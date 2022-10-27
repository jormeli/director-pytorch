import numpy as np
import torch
import torch.optim as optim
import os

from dreamerv2.utils.module import get_parameters, FreezeParameters
from dreamerv2.models.dense import DenseModel
from dreamerv2.models.rssm import RSSM
from dreamerv2.models.pixel import ObsDecoder, ObsEncoder
from dreamerv2.utils.buffer import TransitionBuffer
from director.config import ExperimentConfig
from director.policy import Manager, Worker
from director.goal_autoencoder import GoalAutoencoder
from director.utils import (
    VectorOfCategoricals,
    EMA,
    compute_returns,
    max_cosine_similarity,
)


class Trainer(object):
    def __init__(
        self,
        config,
        device,
        cfg: ExperimentConfig,
    ):
        self.cfg = cfg
        config.rssm_type = "continuous"

        self.goal_vae_latents, self.goal_vae_classes = 8, 8
        self.goal_duration = 8
        self.worker_ext_reward = False
        self.device = device
        self.config = config
        self.action_size = config.action_size
        self.pixel = config.pixel
        self.kl_info = config.kl
        self.seq_len = config.seq_len
        self.batch_size = config.batch_size
        self.collect_intervals = config.collect_intervals
        self.seed_steps = config.seed_steps
        self.discount = config.discount_
        self.lambda_ = config.lambda_
        self.horizon = config.horizon
        self.loss_scale = config.loss_scale
        self.actor_entropy_scale = config.actor_entropy_scale
        self.grad_clip_norm = config.grad_clip

        self._model_initialize(config)
        self._optim_initialize(config)
        self.manager_int_ema = EMA(alpha=0.001)
        self.manager_ext_ema = EMA(alpha=0.001)

    def collect_seed_episodes(self, env):
        print("Collecting seed steps...")
        s, done = env.reset(), False
        for i in range(self.seed_steps):
            a = env.action_space.sample()
            ns, r, done, _ = env.step(a)
            if done:
                self.buffer.add(s, a, r, done)
                s, done = env.reset(), False
            else:
                self.buffer.add(s, a, r, done)
                s = ns
        print("Done.")

    def train(self, train_metrics):
        _model_loss = []
        _goal_vae_loss = []
        _manager_actor_loss = []
        _manager_ext_loss = []
        _manager_int_loss = []
        _worker_actor_loss = []
        _worker_value_loss = []
        _worker_reward = []
        _obs_loss = []
        _goal_norms = []
        _imag_modelstate_norms = []
        for i in range(self.collect_intervals):
            obs, actions, rewards, terms = self.buffer.sample()
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)  # t, t+seq_len
            actions = torch.tensor(actions, dtype=torch.float32).to(
                self.device
            )  # t-1, t+seq_len-1
            rewards = (
                torch.tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(-1)
            )  # t-1 to t+seq_len-1
            nonterms = (
                torch.tensor(1 - terms, dtype=torch.float32)
                .to(self.device)
                .unsqueeze(-1)
            )  # t-1 to t+seq_len-1

            (
                model_loss,
                kl_loss,
                obs_loss,
                reward_loss,
                pcont_loss,
                prior_dist,
                post_dist,
                posterior,
            ) = self.representation_loss(obs, actions, rewards, nonterms)
            _model_loss.append(model_loss.detach().cpu().item())
            _obs_loss.append(obs_loss.detach().cpu().item())

            self.model_optimizer.zero_grad()
            model_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                get_parameters(self.world_list), self.grad_clip_norm
            )
            self.model_optimizer.step()

            post_modelstate = self.RSSM.get_model_state(posterior).detach()
            goal_embedding, goal_logit = self.GoalVAE.encode(post_modelstate)
            goal_reconstruction, reconstruction_dist = self.GoalVAE.decode(
                goal_embedding
            )
            goal_losses = self.GoalVAE.loss(
                post_modelstate,
                reconstruction_dist,
                goal_logit,
            )
            goal_loss = goal_losses["loss"]
            self.goal_vae_optimizer.zero_grad()
            goal_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.GoalVAE.parameters(), self.grad_clip_norm
            )
            self.goal_vae_optimizer.step()
            _goal_vae_loss.append(goal_loss.detach().cpu().item())
            (
                manager_ext_loss,
                manager_int_loss,
                manager_actor_loss,
                worker_value_loss,
                worker_actor_loss,
                worker_reward,
                goals,
                imag_modelstates,
            ) = self.manager_actorcritic_loss(posterior)
            _manager_ext_loss.append(manager_ext_loss.detach().cpu().item())
            _manager_int_loss.append(manager_int_loss.detach().cpu().item())
            _manager_actor_loss.append(manager_actor_loss.detach().cpu().item())
            _worker_actor_loss.append(worker_actor_loss.detach().cpu().item())
            _worker_value_loss.append(worker_value_loss.detach().cpu().item())
            _worker_reward.append(worker_reward.detach().cpu().sum(dim=0).mean().item())
            _goal_norms.append(torch.norm(goals, dim=-1).mean().cpu().item())
            _imag_modelstate_norms.append(
                torch.norm(imag_modelstates, dim=-1).mean().cpu().item()
            )

            self.manager_ext_value_optimizer.zero_grad()
            manager_ext_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.manager.extrinsic_critic.parameters(), self.grad_clip_norm
            )
            self.manager_ext_value_optimizer.step()

            self.manager_int_value_optimizer.zero_grad()
            manager_int_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.manager.intrinsic_critic.parameters(), self.grad_clip_norm
            )
            self.manager_int_value_optimizer.step()

            self.manager_actor_optimizer.zero_grad()
            manager_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.manager.actor.parameters(), self.grad_clip_norm
            )
            self.manager_actor_optimizer.step()

            self.worker_value_optimizer.zero_grad()
            worker_value_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.worker.critic.parameters(), self.grad_clip_norm
            )
            self.worker_value_optimizer.step()

            self.worker_actor_optimizer.zero_grad()
            worker_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.worker.actor.parameters(), self.grad_clip_norm
            )
            self.worker_actor_optimizer.step()

        train_metrics["model_loss"] = np.mean(np.array(_model_loss))
        train_metrics["goal_vae_loss"] = np.mean(_goal_vae_loss)
        train_metrics["manager_actor_loss"] = np.mean(_manager_actor_loss)
        train_metrics["manager_ext_loss"] = np.mean(_manager_ext_loss)
        train_metrics["manager_int_loss"] = np.mean(_manager_int_loss)
        train_metrics["worker_actor_loss"] = np.mean(_worker_actor_loss)
        train_metrics["worker_value_loss"] = np.mean(_worker_value_loss)
        train_metrics["obs_loss"] = np.mean(_obs_loss)
        train_metrics["worker_reward"] = np.mean(_worker_reward)
        train_metrics["imag_model_state_norm"] = np.mean(_imag_modelstate_norms)
        train_metrics["goal_norm"] = np.mean(_goal_norms)

        return train_metrics

    def rollout_imagination(
        self,
        horizon: int,
        prev_rssm_state,
    ):
        def _get_goal(model_state):
            goal_enc, logits = self.manager.get_action(
                self.RSSM.get_model_state(model_state).detach()
            )
            goal_dist = VectorOfCategoricals.get_dist(logits, independent=True)
            goal, _ = self.GoalVAE.decode(goal_enc)
            log_prob = goal_dist.log_prob(
                torch.round(
                    goal_enc.detach().view(
                        -1, self.goal_vae_latents, self.goal_vae_classes
                    )
                )
            )
            entropy = goal_dist.entropy().sum(dim=-1)
            return goal, log_prob, entropy

        rssm_state = prev_rssm_state
        next_rssm_states = []
        action_entropy = []
        imag_log_probs = []
        goal_log_probs = []
        goal_entropy = []
        goals = []

        for t in range(horizon):
            if t % self.goal_duration == 0:
                goal, goal_log_prob, _goal_entropy = _get_goal(rssm_state)

            action_dist = self.worker.get_action(
                self.RSSM.get_model_state(rssm_state).detach(),
                goal.detach(),
            )

            action = action_dist.rsample()
            rssm_state = self.RSSM.rssm_imagine(action, rssm_state)
            next_rssm_states.append(rssm_state)
            action_entropy.append(action_dist.entropy())
            imag_log_probs.append(action_dist.log_prob(torch.round(action.detach())))
            goals.append(goal)
            goal_log_probs.append(goal_log_prob)
            goal_entropy.append(_goal_entropy)
        goals.append(goal)

        next_rssm_states = self.RSSM.rssm_stack_states(next_rssm_states, dim=0)
        action_entropy = torch.stack(action_entropy, dim=0)
        imag_log_probs = torch.stack(imag_log_probs, dim=0)
        goal_log_probs = torch.stack(goal_log_probs, dim=0)
        goal_entropy = torch.stack(goal_entropy, dim=0)
        goals = torch.stack(goals, dim=0)
        return (
            next_rssm_states,
            imag_log_probs,
            goal_log_probs,
            action_entropy,
            goal_entropy,
            goals,
        )

    def manager_actorcritic_loss(self, posterior):
        with torch.no_grad():
            batched_posterior = self.RSSM.rssm_detach(
                self.RSSM.rssm_seq_to_batch(
                    posterior, self.batch_size, self.seq_len - 1
                )
            )

        with FreezeParameters(self.world_list + [self.GoalVAE]):
            (
                imag_rssm_states,
                imag_log_prob,
                goal_log_prob,
                policy_entropy,
                goal_entropy,
                goals,
            ) = self.rollout_imagination(self.horizon, batched_posterior)

        imag_modelstates = self.RSSM.get_model_state(imag_rssm_states)
        with FreezeParameters(
            self.world_list + [self.GoalVAE] + self.value_list + [self.DiscountModel]
        ):
            imag_reward_dist = self.RewardDecoder(imag_modelstates)
            imag_reward = imag_reward_dist.mean
            manager_modelstates = imag_modelstates[:: self.goal_duration]
            manager_reward = torch.cat(
                [
                    chunk.sum(dim=0, keepdim=True)
                    for chunk in torch.split(imag_reward, self.goal_duration, dim=0)
                ],
                dim=0,
            )
            manager_goals = goals[: -1 : self.goal_duration]
            manager_ext_value_dist = self.manager.extrinsic_critic.target_value(
                manager_modelstates
            )
            manager_int_value_dist = self.manager.intrinsic_critic.target_value(
                manager_modelstates
            )
            manager_ext_value = manager_ext_value_dist.mean
            manager_int_value = manager_int_value_dist.mean

            worker_value_dist = self.worker.critic.target_value(
                torch.cat(
                    [
                        imag_modelstates,
                        goals[
                            1:
                        ],  # first goal corresponds to the observed posterior model state
                    ],
                    dim=-1,
                )
            )
            worker_value = worker_value_dist.mean
            discount_dist = self.DiscountModel(imag_modelstates)
            discount_arr = self.discount * torch.round(
                discount_dist.base_dist.probs
            )  # mean = prob(disc==1)
            rr_goals, rr_logits = self.GoalVAE.encode(manager_modelstates)
            _, rr_dist = self.GoalVAE.decode(rr_goals)
            reconstruction_reward = -rr_dist.log_prob(manager_modelstates).unsqueeze(-1)

        manager_discount_arr = discount_arr[:: self.goal_duration]
        manager_ext_returns = compute_returns(
            manager_reward[:-1],
            manager_ext_value[:-1],
            manager_discount_arr[:-1],
            manager_ext_value[-1],
            self.lambda_,
        )
        manager_int_returns = compute_returns(
            reconstruction_reward[:-1].detach(),
            manager_int_value[:-1],
            manager_discount_arr[:-1],
            manager_int_value[-1],
            self.lambda_,
        )
        self.manager_ext_ema.update(
            torch.std(manager_ext_returns.view(-1), dim=0).detach().item(),
            manager_ext_returns.mean().detach().item(),
        )
        self.manager_int_ema.update(
            torch.std(manager_int_returns.view(-1), dim=0).detach().item(),
            manager_int_returns.mean().detach().item(),
        )
        manager_ext_returns = (
            manager_ext_returns - self.manager_ext_ema.loc
        ) / self.manager_ext_ema.scale + self.manager_ext_ema.loc
        manager_int_returns = (
            manager_int_returns - self.manager_int_ema.loc
        ) / self.manager_int_ema.scale + self.manager_int_ema.loc
        manager_ext_value = (
            manager_ext_value - self.manager_ext_ema.loc
        ) / self.manager_ext_ema.scale + self.manager_ext_ema.loc
        manager_int_value = (
            manager_int_value - self.manager_int_ema.loc
        ) / self.manager_int_ema.scale + self.manager_int_ema.loc

        manager_ext_loss = self.manager.extrinsic_critic_loss(
            manager_modelstates,
            manager_discount_arr,
            manager_ext_returns,
        )

        manager_int_loss = self.manager.intrinsic_critic_loss(
            manager_modelstates,
            manager_discount_arr,
            manager_int_returns,
        )

        manager_actor_loss = self.manager.actor_loss(
            manager_ext_returns,
            manager_int_returns,
            manager_discount_arr,
            goal_entropy[:: self.goal_duration],
            manager_ext_value,
            manager_int_value,
            goal_log_prob[:: self.goal_duration],
        )

        worker_reward = max_cosine_similarity(
            goals[:-1].detach(),
            imag_modelstates.detach(),
            dim=-1,
        )
        if self.worker_ext_reward:
            worker_reward = worker_reward + imag_reward

        worker_value_losses = []
        worker_actor_losses = []

        # Split worker trajectories by goal
        for idx in range(0, self.horizon, self.goal_duration):
            idxs = slice(idx, idx + self.goal_duration)
            traj_reward = worker_reward[idxs]
            traj_value = worker_value[idxs]
            traj_discount_arr = discount_arr[idxs]
            worker_returns = compute_returns(
                traj_reward[:-1],
                traj_value[:-1],
                traj_discount_arr[:-1],
                traj_value[-1],
                self.lambda_,
            ).detach()
            worker_value_losses.append(
                self.worker.value_loss(
                    torch.cat(
                        [
                            imag_modelstates[idxs],
                            goals[1:][idxs],
                        ],
                        dim=-1,
                    ),
                    traj_discount_arr,
                    worker_returns,
                )
            )
            traj_policy_entropy = policy_entropy[idxs]
            traj_imag_log_prob = imag_log_prob[idxs]
            worker_actor_losses.append(
                self.worker.actor_loss(
                    worker_returns,
                    traj_discount_arr,
                    traj_policy_entropy,
                    traj_value,
                    traj_imag_log_prob,
                )
            )

        return (
            manager_ext_loss,
            manager_int_loss,
            manager_actor_loss,
            sum(worker_value_losses),
            sum(worker_actor_losses),
            worker_reward,
            goals.detach(),
            imag_modelstates.detach(),
        )

    # Copied from the DreamerV2 implementation
    def representation_loss(self, obs, actions, rewards, nonterms):
        embed = self.ObsEncoder(obs)  # t to t+seq_len
        prev_rssm_state = self.RSSM._init_rssm_state(self.batch_size)
        prior, posterior = self.RSSM.rollout_observation(
            self.seq_len, embed, actions, nonterms, prev_rssm_state
        )
        post_modelstate = self.RSSM.get_model_state(posterior)  # t to t+seq_len
        obs_dist = self.ObsDecoder(post_modelstate[:-1])  # t to t+seq_len-1
        reward_dist = self.RewardDecoder(post_modelstate[:-1])  # t to t+seq_len-1
        pcont_dist = self.DiscountModel(post_modelstate[:-1])  # t to t+seq_len-1

        obs_loss = self._obs_loss(obs_dist, obs[:-1])
        reward_loss = self._reward_loss(reward_dist, rewards[1:])
        pcont_loss = self._pcont_loss(pcont_dist, nonterms[1:])
        prior_dist, post_dist, div = self._kl_loss(prior, posterior)

        model_loss = (
            self.loss_scale["kl"] * div
            + reward_loss
            + obs_loss
            + self.loss_scale["discount"] * pcont_loss
        )
        return (
            model_loss,
            div,
            obs_loss,
            reward_loss,
            pcont_loss,
            prior_dist,
            post_dist,
            posterior,
        )

    # Copied from the DreamerV2 implementation
    def _obs_loss(self, obs_dist, obs):
        obs_loss = -torch.mean(obs_dist.log_prob(obs))
        return obs_loss

    # Copied from the DreamerV2 implementation
    def _kl_loss(self, prior, posterior):
        prior_dist = self.RSSM.get_dist(prior)
        post_dist = self.RSSM.get_dist(posterior)
        if self.kl_info["use_kl_balance"]:
            alpha = self.kl_info["kl_balance_scale"]
            kl_lhs = torch.mean(
                torch.distributions.kl.kl_divergence(
                    self.RSSM.get_dist(self.RSSM.rssm_detach(posterior)), prior_dist
                )
            )
            kl_rhs = torch.mean(
                torch.distributions.kl.kl_divergence(
                    post_dist, self.RSSM.get_dist(self.RSSM.rssm_detach(prior))
                )
            )
            if self.kl_info["use_free_nats"]:
                free_nats = self.kl_info["free_nats"]
                kl_lhs = torch.max(kl_lhs, kl_lhs.new_full(kl_lhs.size(), free_nats))
                kl_rhs = torch.max(kl_rhs, kl_rhs.new_full(kl_rhs.size(), free_nats))
            kl_loss = alpha * kl_lhs + (1 - alpha) * kl_rhs

        else:
            kl_loss = torch.mean(
                torch.distributions.kl.kl_divergence(post_dist, prior_dist)
            )
            if self.kl_info["use_free_nats"]:
                free_nats = self.kl_info["free_nats"]
                kl_loss = torch.max(
                    kl_loss, kl_loss.new_full(kl_loss.size(), free_nats)
                )
        return prior_dist, post_dist, kl_loss

    # Copied from the DreamerV2 implementation
    def _reward_loss(self, reward_dist, rewards):
        reward_loss = -torch.mean(reward_dist.log_prob(rewards))
        return reward_loss

    # Copied from the DreamerV2 implementation
    def _pcont_loss(self, pcont_dist, nonterms):
        pcont_target = nonterms.float()
        pcont_loss = -torch.mean(pcont_dist.log_prob(pcont_target))
        return pcont_loss

    def update_target(self):
        self.manager.intrinsic_critic.update_target()
        self.manager.extrinsic_critic.update_target()
        self.worker.critic.update_target()

    def save_model(self, iter):
        save_dict = self.get_save_dict()
        model_dir = self.config.model_dir
        save_path = os.path.join(model_dir, "models_%d.pth" % iter)
        torch.save(save_dict, save_path)

    def get_save_dict(self):
        return {
            "RSSM": self.RSSM.state_dict(),
            "ObsEncoder": self.ObsEncoder.state_dict(),
            "ObsDecoder": self.ObsDecoder.state_dict(),
            "RewardDecoder": self.RewardDecoder.state_dict(),
            "Manager": self.manager.state_dict(),
            "Worker": self.worker.state_dict(),
            "DiscountModel": self.DiscountModel.state_dict(),
            "GoalVAE": self.GoalVAE.state_dict(),
        }

    def load_save_dict(self, saved_dict):
        self.RSSM.load_state_dict(saved_dict["RSSM"])
        self.ObsEncoder.load_state_dict(saved_dict["ObsEncoder"])
        self.ObsDecoder.load_state_dict(saved_dict["ObsDecoder"])
        self.RewardDecoder.load_state_dict(saved_dict["RewardDecoder"])
        self.ActionModel.load_state_dict(saved_dict["ActionModel"])
        self.ValueModel.load_state_dict(saved_dict["ValueModel"])
        self.DiscountModel.load_state_dict(saved_dict["DiscountModel"])

    def _model_initialize(self, config):
        obs_shape = config.obs_shape
        action_size = config.action_size
        deter_size = config.rssm_info["deter_size"]
        if config.rssm_type == "continuous":
            stoch_size = config.rssm_info["stoch_size"]
        elif config.rssm_type == "discrete":
            category_size = config.rssm_info["category_size"]
            class_size = config.rssm_info["class_size"]
            stoch_size = category_size * class_size

        embedding_size = config.embedding_size
        rssm_node_size = config.rssm_node_size
        modelstate_size = stoch_size + deter_size

        self.buffer = TransitionBuffer(
            config.capacity,
            obs_shape,
            action_size,
            config.seq_len,
            config.batch_size,
            config.obs_dtype,
            config.action_dtype,
        )
        self.RSSM = RSSM(
            action_size,
            rssm_node_size,
            embedding_size,
            self.device,
            config.rssm_type,
            config.rssm_info,
        ).to(self.device)
        self.GoalVAE = GoalAutoencoder(
            config.rssm_info["deter_size"] + config.rssm_info["stoch_size"],
            self.goal_vae_latents,
            self.goal_vae_classes,
        ).to(self.device)
        self.RewardDecoder = DenseModel((1,), modelstate_size, config.reward).to(
            self.device
        )
        self.manager = Manager(
            config.rssm_info["deter_size"] + config.rssm_info["stoch_size"],
            self.goal_vae_latents,
            self.goal_vae_classes,
            n_layers=4,
            layer_size=512,
            slow_target_mix=self.cfg.manager_cfg.slow_target_mix,
        ).to(self.device)
        self.worker = Worker(
            2 * (config.rssm_info["deter_size"] + config.rssm_info["stoch_size"]),
            action_size,
            n_layers=4,
            layer_size=512,
            slow_target_mix=self.cfg.worker_cfg.slow_target_mix,
        ).to(self.device)

        if config.discount["use"]:
            self.DiscountModel = DenseModel((1,), modelstate_size, config.discount).to(
                self.device
            )
        if config.pixel:
            self.ObsEncoder = ObsEncoder(
                obs_shape, embedding_size, config.obs_encoder
            ).to(self.device)
            self.ObsDecoder = ObsDecoder(
                obs_shape, modelstate_size, config.obs_decoder
            ).to(self.device)
        else:
            self.ObsEncoder = DenseModel(
                (embedding_size,), int(np.prod(obs_shape)), config.obs_encoder
            ).to(self.device)
            self.ObsDecoder = DenseModel(
                obs_shape, modelstate_size, config.obs_decoder
            ).to(self.device)

    def _optim_initialize(self, config):
        model_lr = config.lr["model"]
        actor_lr = 1e-4  # config.lr["actor"]
        value_lr = config.lr["critic"]
        self.world_list = [
            self.ObsEncoder,
            self.RSSM,
            self.RewardDecoder,
            self.ObsDecoder,
            self.DiscountModel,
        ]
        self.actor_list = [
            self.manager.actor,
            self.worker.actor,
        ]
        self.value_list = [
            self.manager.extrinsic_critic,
            self.manager.intrinsic_critic,
            self.worker.critic,
        ]
        self.actorcritic_list = [
            *self.actor_list,
            *self.value_list,
        ]
        self.model_optimizer = optim.Adam(get_parameters(self.world_list), model_lr)
        self.manager_actor_optimizer = optim.Adam(
            self.manager.actor.parameters(),
            lr=actor_lr,
            weight_decay=1e-2,
            eps=1e-6,
        )
        self.manager_ext_value_optimizer = optim.Adam(
            self.manager.extrinsic_critic.parameters(),
            lr=value_lr,
            weight_decay=1e-2,
            eps=1e-6,
        )
        self.manager_int_value_optimizer = optim.Adam(
            self.manager.intrinsic_critic.parameters(),
            lr=value_lr,
            weight_decay=1e-2,
            eps=1e-6,
        )
        self.worker_actor_optimizer = optim.Adam(
            self.worker.actor.parameters(),
            lr=actor_lr,
            weight_decay=1e-2,
            eps=1e-6,
        )
        self.worker_value_optimizer = optim.Adam(
            self.worker.critic.parameters(),
            lr=value_lr,
            weight_decay=1e-2,
            eps=1e-6,
        )
        self.goal_vae_optimizer = optim.Adam(
            self.GoalVAE.parameters(),
            lr=1e-4,
            weight_decay=1e-2,
            eps=1e-6,
        )

    def _print_summary(self):
        print("\n Obs encoder: \n", self.ObsEncoder)
        print("\n RSSM model: \n", self.RSSM)
        print("\n Reward decoder: \n", self.RewardDecoder)
        print("\n Obs decoder: \n", self.ObsDecoder)
        if self.config.discount["use"]:
            print("\n Discount decoder: \n", self.DiscountModel)
        print("\n Actor: \n", self.ActionModel)
        print("\n Critic: \n", self.ValueModel)
