from datetime import datetime
import wandb
import logging
import os
import torch
import numpy as np
import gym
from ray.rllib.env import VectorEnv
from skimage.transform import resize as sk_resize
from typing import Optional, Tuple

from dreamerv2.utils.wrapper import (
    GymMinAtar,
    OneHotAction,
    breakoutPOMDP,
    space_invadersPOMDP,
    seaquestPOMDP,
    asterixPOMDP,
    freewayPOMDP,
)
from dreamerv2.training.config import MinAtarConfig

from director.config import ExperimentConfig, Device
from director.log import configure_logging
from director.trainer import Trainer

import hydra
from omegaconf import OmegaConf
from torchvision.utils import make_grid
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

pomdp_wrappers = {
    "breakout": breakoutPOMDP,
    "seaquest": seaquestPOMDP,
    "space_invaders": space_invadersPOMDP,
    "asterix": asterixPOMDP,
    "freeway": freewayPOMDP,
}


class TransposeWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        resize: Optional[Tuple[int]] = None,
    ):
        super().__init__(env)
        self.env = env
        self.resize = resize
        obs_shape = env.observation_space.shape
        low, high = env.observation_space.low, env.observation_space.high
        if len(obs_shape) < 3:
            obs_shape = obs_shape + (1,)
            low = low[..., None]
            high = high[..., None]
        if resize is not None:
            obs_shape = tuple(self.resize) + (obs_shape[-1],)
            low = sk_resize(low, resize)
            high = sk_resize(high, resize)
        self.observation_space = gym.spaces.Box(
            low=low.transpose(2, 0, 1),
            high=high.transpose(2, 0, 1),
            shape=(obs_shape[-1],) + obs_shape[:-1],
            dtype=env.observation_space.dtype,
        )

    def _transform_obs(self, obs):
        if len(obs.shape) < 3:
            obs = obs[..., None]
        if self.resize:
            obs = sk_resize(obs, self.resize)
        return obs.transpose(2, 0, 1)

    def step(self, *args, **kwargs):
        next_state, reward, done, info = self.env.step(*args, **kwargs)
        return (self._transform_obs(next_state), reward, done, info)

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        return self._transform_obs(obs)


def make_env(cfg):
    def _create_minatar(env_name):
        def _create_env(*args):
            return OneHotAction(GymMinAtar(env_name))

        return _create_env

    if cfg.environment_cfg.name.startswith("minatar:"):
        env_name = cfg.environment_cfg.name.split(":")[1]
        env = VectorEnv.vectorize_gym_envs(
            make_env=_create_minatar(env_name),
            num_envs=cfg.environment_cfg.num_parallel_envs,
        )
        action_size = env.action_space.shape[0]
        obs_shape = env.observation_space.shape
    elif cfg.environment_cfg.name.startswith("gym:"):
        def _create_env(*args, **kwargs):
            return OneHotAction(
                TransposeWrapper(
                    gym.make(env_name, **cfg.environment_cfg.env_args),
                    resize=cfg.environment_cfg.resize_obs,
                )
            )
        env_name = cfg.environment_cfg.name.split(":")[1]
        env = VectorEnv.vectorize_gym_envs(
            make_env=_create_env,
            num_envs=cfg.environment_cfg.num_parallel_envs,
        )
        action_size = env.action_space.shape[0]
        if cfg.environment_cfg.observation_shape is not None:
            obs_shape = tuple(cfg.environment_cfg.observation_shape)
        else:
            obs_shape = env.observation_space.shape

    return env, action_size, obs_shape


def action_noise(action, itr):
    return action
    eps = 0.4 - (itr / 1e6) * 0.4
    if np.random.rand() < eps:
        rand_act = torch.zeros_like(action)
        rand_act[:, torch.randint(0, action.shape[-1], (1,))[0]] = 1

        return rand_act.to(action.device)
    return action


def run(cfg):
    configure_logging(use_json=False)
    wandb.login()
    env_name = cfg.environment_cfg.name
    exp_id = datetime.now().isoformat() + "_pomdp"

    result_dir = os.path.join("results", "{}_{}".format(env_name, exp_id))
    model_dir = os.path.join(result_dir, "models")  # dir to save learnt models
    os.makedirs(model_dir, exist_ok=True)
    best_save_path = os.path.join(model_dir, "best_model.pth")

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.device == Device.CUDA:
        assert torch.cuda.is_available()

    device = torch.device(cfg.device)
    logging.info(f"Using device: {device}")

    env, action_size, obs_shape = make_env(cfg)
    num_parallel_envs = cfg.environment_cfg.num_parallel_envs
    batch_size = cfg.training_cfg.batch_size
    seq_len = cfg.training_cfg.sequence_length

    config = MinAtarConfig(
        env=env_name,
        obs_shape=obs_shape,
        action_size=action_size,
        obs_dtype=None,
        action_dtype=None,
        seq_len=seq_len,
        batch_size=batch_size,
        model_dir=model_dir,
    )
    config.train_every = cfg.training_cfg.train_every
    config.train_steps = cfg.training_cfg.train_steps
    config.batch_size = cfg.training_cfg.batch_size
    config.horizon = cfg.training_cfg.horizon
    config.seed_steps = cfg.training_cfg.seed_steps
    config.seq_len = cfg.training_cfg.sequence_length
    config.slow_target_update = cfg.training_cfg.slow_target_update
    goal_duration = cfg.training_cfg.goal_duration

    config_dict = config.__dict__
    trainer = Trainer(config, device, cfg)
    trainer.goal_duration = goal_duration

    with wandb.init(project="Director", config=config_dict):
        logging.info("Training...")
        trainer.collect_seed_episodes(env)
        obs, score = env.vector_reset(), np.zeros(num_parallel_envs)
        prev_rssmstate = trainer.RSSM._init_rssm_state(num_parallel_envs)
        prev_action = torch.zeros(num_parallel_envs, trainer.action_size).to(trainer.device)
        scores = []
        best_mean_score = -1e10
        train_episodes = 0
        goal_iteration = np.zeros((num_parallel_envs,), dtype=np.int32)
        goal = torch.zeros(
            num_parallel_envs,
            config.rssm_info["deter_size"] + config.rssm_info["stoch_size"],
        ).to(device)

        # Histories used for visualizations
        grid = None
        progress = tqdm.tqdm(total=trainer.config.train_steps, desc="Training")
        should_update_stats = False
        with logging_redirect_tqdm():
            for iter in range(1, trainer.config.train_steps):
                train_metrics = {}
                obs = np.array(obs).astype(np.float32)
                if iter % cfg.training_cfg.train_every == 0:
                    train_metrics = trainer.train(train_metrics)
                    should_update_stats = True
                if (
                    iter
                    % (cfg.training_cfg.slow_target_update * cfg.training_cfg.train_every)
                    == 0
                ):
                    trainer.update_target()
                if (iter - 1) % cfg.training_cfg.save_every == 0:
                    trainer.save_model(iter)
                with torch.no_grad():
                    embed = trainer.ObsEncoder(
                        torch.tensor(obs, dtype=torch.float32)
                        .to(trainer.device)
                    )
                    _, posterior_rssm_state = trainer.RSSM.rssm_observe(
                        embed, prev_action, True, prev_rssmstate
                    )
                    model_state = trainer.RSSM.get_model_state(posterior_rssm_state)
                    goal_update_idxs = goal_iteration % goal_duration == 0
                    if goal_update_idxs.sum() > 0:
                        env_goal, _ = trainer.manager.get_action(
                            model_state[goal_update_idxs]
                        )
                        env_goal, _ = trainer.GoalVAE.decode(env_goal)
                        goal[goal_update_idxs] = env_goal

                    if (iter - 1) % 1000 == 0:
                        # Log images
                        trainer.ObsDecoder.eval()
                        trainer.GoalVAE.eval()
                        episode = trainer.buffer._storage[-1]
                        observations = torch.tensor(episode["obs"][-8:]).float().to(trainer.device)
                        trainer.ObsDecoder.train()
                        trainer.GoalVAE.train()
                        grid = make_grid(
                            torch.cat(
                                [
                                    observations,
                                ],
                                dim=0
                            ),
                            nrow=8,
                            padding=2,
                        )
                        if grid.shape[0] > 3:
                            grid = grid[
                                [0, 1, 3]
                            ]  # breakout env has 4 channels, remove the "trail" channel for visualizations

                    action, action_dist = trainer.worker.get_action(
                        model_state,
                        goal,
                    )

                next_obs, rew, done, _ = env.vector_step(
                    action.cpu().numpy()
                )
                score += rew

                goal_iteration += 1
                for worker_id in range(num_parallel_envs):
                    trainer.buffer.add(
                        obs[worker_id],
                        action[worker_id].cpu().numpy(),
                        rew[worker_id],
                        done[worker_id],
                        worker_id
                    )

                for env_id in range(num_parallel_envs):
                    if done[env_id]:
                        should_update_stats = True
                        progress.update(goal_iteration[env_id])
                        goal_iteration[env_id] = 0
                        w_obs = env.reset_at(env_id)
                        obs[env_id] = w_obs
                        new_rssm_state = trainer.RSSM._init_rssm_state(1)
                        prev_rssmstate.mean[env_id] = new_rssm_state.mean[0]
                        prev_rssmstate.std[env_id] = new_rssm_state.std[0]
                        prev_rssmstate.stoch[env_id] = new_rssm_state.stoch[0]
                        prev_rssmstate.deter[env_id] = new_rssm_state.deter[0]
                        prev_action[env_id] = torch.zeros(trainer.action_size).to(trainer.device)
                        train_episodes += 1
                        train_metrics["train_rewards"] = max(
                            score[env_id],
                            train_metrics.get("train_rewards", score[env_id]),
                        )
                        train_metrics["train_steps"] = iter * num_parallel_envs
                        train_metrics["train_episodes"] = train_episodes
                        if grid is not None:
                            train_metrics["frames"] = wandb.Image(
                                grid,
                                caption=(
                                    "1: Observation, "
                                    + "2: Reconstructed observation, "
                                    + "3: Reconstructed GoalVAE observation, "
                                    + "4: Goal"
                                ),
                            )
                            grid = None

                        progress.set_description(
                            f"Training | Episode = {train_episodes} / Reward = {train_metrics['train_rewards']:.2f} / Best = {best_mean_score:2.2f}"
                        )
                        scores.append(score[env_id])
                        score[env_id] = 0.
                        if len(scores) > 100:
                            scores.pop(0)
                            current_average = np.mean(scores)
                            if current_average > best_mean_score:
                                best_mean_score = current_average
                                logging.info(f"Saving best model with mean score: {best_mean_score}")
                                save_dict = trainer.get_save_dict()
                                torch.save(save_dict, best_save_path)
                    else:
                        obs[env_id] = next_obs[env_id]
                        prev_rssmstate.mean[env_id] = posterior_rssm_state.mean[env_id]
                        prev_rssmstate.std[env_id] = posterior_rssm_state.std[env_id]
                        prev_rssmstate.stoch[env_id] = posterior_rssm_state.stoch[env_id]
                        prev_rssmstate.deter[env_id] = posterior_rssm_state.deter[env_id]
                        prev_action[env_id] = action[env_id]

                    if should_update_stats:
                        wandb.log(train_metrics, step=iter)
                        should_update_stats = False


@hydra.main(
    version_base=None, config_path="./experiment", config_name="experiment_config"
)
def main(cfg: ExperimentConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    run(cfg)


if __name__ == "__main__":
    main()
