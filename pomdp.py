from datetime import datetime
import wandb
import os
import torch
import numpy as np
import gym
from PIL import Image

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
from director.trainer import Trainer
from director.utils import VectorOfCategoricals

import hydra
from omegaconf import OmegaConf
from torchvision.utils import make_grid

pomdp_wrappers = {
    "breakout": breakoutPOMDP,
    "seaquest": seaquestPOMDP,
    "space_invaders": space_invadersPOMDP,
    "asterix": asterixPOMDP,
    "freeway": freewayPOMDP,
}


class TransposeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def _transform_obs(self, obs):
        shape = obs.shape
        img = Image.fromarray(obs)
        img.thumbnail((shape[0] // 2, shape[1] // 2))
        return np.asarray(img).transpose(2, 0, 1)

    def step(self, *args, **kwargs):
        next_state, reward, terminated, info, done = self.env.step(*args, **kwargs)
        return (self._transform_obs(next_state), reward, done or terminated, info)

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        return self._transform_obs(obs)


def make_env(cfg):
    if cfg.environment.startswith("minatar:"):
        env_name = cfg.environment.split(":")[1]
        # PomdpWrapper = pomdp_wrappers[env_name]
        # env = PomdpWrapper(OneHotAction(GymMinAtar(env_name)))
        env = OneHotAction(GymMinAtar(env_name))
        obs_shape = env.observation_space.shape
        action_size = env.action_space.shape[0]
        obs_dtype = bool
        action_dtype = np.float32
    elif cfg.environment.startswith("gym:"):
        env_name = cfg.environment.split(":")[1]
        env = OneHotAction(TransposeWrapper(gym.make(env_name, continuous=False)))
        action_size = env.action_space.shape[0]
        # obs_shape = tuple(np.array(env.observation_space.shape)[[2,0,1]].tolist())
        obs_shape = (3, 48, 48)
        obs_dtype = env.observation_space.dtype
        action_dtype = np.float32

    return env, obs_shape, action_size, obs_dtype, action_dtype


def action_noise(action, itr):
    return action
    eps = 0.4 - (itr / 1e6) * 0.4
    if np.random.rand() < eps:
        rand_act = torch.zeros_like(action)
        rand_act[:, torch.randint(0, action.shape[-1], (1,))[0]] = 1

        return rand_act.to(action.device)
    return action


def run(cfg):
    wandb.login()
    env_name = cfg.environment
    exp_id = datetime.now().isoformat() + "_pomdp"

    result_dir = os.path.join("results", "{}_{}".format(env_name, exp_id))
    model_dir = os.path.join(result_dir, "models")  # dir to save learnt models
    os.makedirs(model_dir, exist_ok=True)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.device == Device.CUDA:
        assert torch.cuda.is_available()

    device = torch.device(cfg.device)
    print("Using device:", device)

    env, obs_shape, action_size, obs_dtype, action_dtype = make_env(cfg)
    batch_size = cfg.training_cfg.batch_size
    seq_len = cfg.training_cfg.sequence_length

    config = MinAtarConfig(
        env=env_name,
        obs_shape=obs_shape,
        action_size=action_size,
        obs_dtype=obs_dtype,
        action_dtype=action_dtype,
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
    trainer = Trainer(config, device)
    trainer.goal_duration = goal_duration

    with wandb.init(project="Director", config=config_dict):
        print("Training...")
        train_metrics = {}
        trainer.collect_seed_episodes(env)
        obs, score = env.reset(), 0
        done = False
        prev_rssmstate = trainer.RSSM._init_rssm_state(1)
        prev_action = torch.zeros(1, trainer.action_size).to(trainer.device)
        episode_actor_ent = []
        episode_goal_ent = []
        scores = []
        best_mean_score = 0
        train_episodes = 0
        goal_iteration = 0  # used to keep track of goal age between episode resets

        # Histories used for visualizations
        goal_hist = []
        state_hist = []
        state_dec_hist = []
        obs_hist = []
        grid = None

        for iter in range(1, trainer.config.train_steps):
            obs = obs.astype(np.float32)
            if iter % cfg.training_cfg.train_every == 0:
                train_metrics = trainer.train(train_metrics)
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
                    .unsqueeze(0)
                    .to(trainer.device)
                )
                _, posterior_rssm_state = trainer.RSSM.rssm_observe(
                    embed, prev_action, not done, prev_rssmstate
                )
                model_state = trainer.RSSM.get_model_state(posterior_rssm_state)
                if goal_iteration % goal_duration == 0:
                    goal, logits = trainer.manager.get_action(model_state)
                    goal_dist = VectorOfCategoricals.get_dist(
                        logits
                    )  # used for logging
                    goal, _ = trainer.GoalVAE.decode(goal)

                # Used for logging
                state_enc, _ = trainer.GoalVAE.encode(model_state)
                state_dec, _ = trainer.GoalVAE.decode(state_enc)
                state_dec_hist.append(state_dec)
                goal_ent = torch.mean(goal_dist.entropy()).item()
                episode_goal_ent.append(goal_ent)
                goal_hist.append(goal.detach())
                state_hist.append(model_state.detach())
                obs_hist.append(torch.tensor(obs).float().unsqueeze(0))
                if len(goal_hist) > 8:
                    goal_hist.pop(0)
                if len(state_hist) > 8:
                    state_hist.pop(0)
                if len(obs_hist) > 8:
                    obs_hist.pop(0)
                if len(state_dec_hist) > 8:
                    state_dec_hist.pop(0)

                if (iter - 1) % 100 == 0:
                    # Log images
                    trainer.ObsDecoder.eval()
                    trainer.GoalVAE.eval()
                    goal_img = trainer.ObsDecoder(torch.cat(goal_hist, dim=0)).mean
                    goal_img_ = trainer.ObsDecoder(torch.cat(state_hist, dim=0)).mean
                    goal_img__ = torch.cat(obs_hist, dim=0).to(trainer.device)
                    goal_img___ = trainer.ObsDecoder(
                        torch.cat(state_dec_hist, dim=0)
                    ).mean
                    trainer.ObsDecoder.train()
                    trainer.GoalVAE.train()
                    grid = make_grid(
                        torch.cat(
                            [goal_img__, goal_img_, goal_img___, goal_img], dim=0
                        ),
                        nrow=8,
                        padding=2,
                    )
                    if grid.shape[0] > 3:
                        grid = grid[
                            [0, 1, 3]
                        ]  # breakout env has 4 channels, remove the "trail" channel for visualizations

                action_dist = trainer.worker.get_action(
                    model_state,
                    goal,
                )
                action = action_dist.sample()

                if cfg.worker_cfg.action_noise:
                    action = action_noise(action, iter)

                action_ent = torch.mean(action_dist.entropy()).item()
                episode_actor_ent.append(action_ent)

            next_obs, rew, done, _ = env.step(action.squeeze(0).cpu().numpy())
            score += rew

            goal_iteration += 1
            if done:
                train_episodes += 1
                goal_iteration = 0
                trainer.buffer.add(obs, action.squeeze(0).cpu().numpy(), rew, done)
                train_metrics["train_rewards"] = score
                train_metrics["action_ent"] = np.mean(episode_actor_ent)
                train_metrics["goal_ent"] = np.mean(episode_goal_ent)
                train_metrics["train_steps"] = iter
                if grid is not None and (train_episodes - 1) % 100 == 0:
                    train_metrics["frames"] = wandb.Image(
                        grid,
                        caption=(
                            "1: Observation, "
                            + "2: Reconstructed observation, "
                            + "3: Reconstructed GoalVAE observation, "
                            + "4: Goal"
                        ),
                    )
                wandb.log(train_metrics, step=train_episodes)
                scores.append(score)
                if len(scores) > 100:
                    scores.pop(0)
                    current_average = np.mean(scores)
                    if current_average > best_mean_score:
                        best_mean_score = current_average
                        print("saving best model with mean score : ", best_mean_score)
                        # save_dict = trainer.get_save_dict()
                        # torch.save(save_dict, best_save_path)

                obs, score = env.reset(), 0
                done = False
                prev_rssmstate = trainer.RSSM._init_rssm_state(1)
                prev_action = torch.zeros(1, trainer.action_size).to(trainer.device)
                episode_actor_ent = []
                episode_goal_ent = []
            else:
                trainer.buffer.add(
                    obs, action.squeeze(0).detach().cpu().numpy(), rew, done
                )
                obs = next_obs
                prev_rssmstate = posterior_rssm_state
                prev_action = action


@hydra.main(
    version_base=None, config_path="./experiment", config_name="experiment_config"
)
def main(cfg: ExperimentConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    run(cfg)


if __name__ == "__main__":
    main()
