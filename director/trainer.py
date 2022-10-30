import numpy as np
import torch
import torch.optim as optim
import os
import logging

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
        self.worker_ext_reward = cfg.worker_cfg.extrinsic_reward
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
        logging.info("Collecting seed steps...")
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
        for i in range(1):
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

            _, posterior = self.RSSM.rollout_observation(
                self.seq_len,
                self.ObsEncoder(obs),
                actions,
                nonterms,
                self.RSSM._init_rssm_state(self.batch_size)
            )

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
            #(
            #    manager_ext_loss,
            #    manager_int_loss,
            #    manager_actor_loss,
            #    worker_value_loss,
            #    worker_actor_loss,
            #    worker_reward,
            #    goals,
            #    imag_modelstates,
            #) = self.director_actorcritic_loss(posterior)
            (
                worker_actor_loss,
                worker_value_loss,
                manager_actor_loss,
                manager_value_loss,
                worker_reward,
            ) = self.director_loss(posterior)
            _manager_ext_loss.append(manager_value_loss.detach().cpu().item())
            #if self.cfg.manager_cfg.intrinsic_reward:
            #    _manager_int_loss.append(manager_int_loss.detach().cpu().item())
            _manager_actor_loss.append(manager_actor_loss.detach().cpu().item())
            _worker_actor_loss.append(worker_actor_loss.detach().cpu().item())
            _worker_value_loss.append(worker_value_loss.detach().cpu().item())
            _worker_reward.append(worker_reward.detach().cpu().sum(dim=0).mean().item())
            #_goal_norms.append(torch.norm(goals, dim=-1).mean().cpu().item())
            #_imag_modelstate_norms.append(
            #    torch.norm(imag_modelstates, dim=-1).mean().cpu().item()
            #)

            self.manager_ext_value_optimizer.zero_grad()
            manager_value_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.manager.extrinsic_critic.parameters(), self.grad_clip_norm
            )
            self.manager_ext_value_optimizer.step()

            #if self.cfg.manager_cfg.intrinsic_reward:
            #    self.manager_int_value_optimizer.zero_grad()
            #    manager_int_loss.backward()
            #    torch.nn.utils.clip_grad_norm_(
            #        self.manager.intrinsic_critic.parameters(), self.grad_clip_norm
            #    )
            #    self.manager_int_value_optimizer.step()

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
        if self.cfg.manager_cfg.intrinsic_reward:
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
        horizon:int,
        prev_rssm_state,
    ):
        def _get_goal(model_state):
            latent_goal, logits = self.manager.get_action(
                self.RSSM.get_model_state(model_state).detach()
            )
            goal_dist = VectorOfCategoricals.get_dist(logits, independent=True)
            goal, _ = self.GoalVAE.decode(latent_goal)
            log_prob = goal_dist.log_prob(
                torch.round(
                    latent_goal.detach().view(
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
        goals = []
        goal_entropies = []
        goal_log_probs = []
        for t in range(horizon):
            if t % self.goal_duration == 0:
                goal, goal_log_prob, goal_entropy = _get_goal(rssm_state)

            action, action_dist = self.worker.get_action(
                self.RSSM.get_model_state(rssm_state).detach(),
                goal.detach(),
            )
            rssm_state = self.RSSM.rssm_imagine(action, rssm_state)
            next_rssm_states.append(rssm_state)
            action_entropy.append(action_dist.entropy())
            imag_log_probs.append(action_dist.log_prob(torch.round(action.detach())))
            goals.append(goal)
            goal_entropies.append(goal_entropy)
            goal_log_probs.append(goal_log_prob)

        next_rssm_states = self.RSSM.rssm_stack_states(next_rssm_states, dim=0)
        action_entropy = torch.stack(action_entropy, dim=0)
        imag_log_probs = torch.stack(imag_log_probs, dim=0)
        goals = torch.stack(goals, dim=0)
        goal_entropies = torch.stack(goal_entropies, dim=0)
        goal_log_probs = torch.stack(goal_log_probs, dim=0)
        return (
            next_rssm_states,
            imag_log_probs,
            action_entropy,
            goals,
            goal_entropies,
            goal_log_probs,
        )

    def director_loss(self, posterior):
        with torch.no_grad():
            batched_posterior = self.RSSM.rssm_detach(
                self.RSSM.rssm_seq_to_batch(posterior, self.batch_size, self.seq_len-1)
            )

        with FreezeParameters(self.world_list):
            (
                imag_rssm_states,
                imag_log_prob,
                policy_entropy,
                goals,
                goal_entropies,
                goal_log_probs,
            ) = self.rollout_imagination(self.horizon, batched_posterior)

        imag_modelstates = self.RSSM.get_model_state(imag_rssm_states).detach()
        manager_modelstates = torch.cat(
            [
                self.RSSM.get_model_state(batched_posterior).unsqueeze(0),
                imag_modelstates,
            ],
            dim=0,
        )[::self.goal_duration]

        with FreezeParameters(self.world_list+self.value_list+[self.DiscountModel]):
            imag_reward_dist = self.RewardDecoder(imag_modelstates)
            imag_reward = imag_reward_dist.mean
            worker_value_dist = self.worker.critic.target_value(imag_modelstates)
            worker_value = worker_value_dist.mean
            manager_value = self.manager.extrinsic_critic.target_value(manager_modelstates).mean
            manager_reward = torch.cat(
                [
                    chunk.sum(dim=0, keepdim=True)
                    for chunk in torch.split(imag_reward, self.goal_duration, dim=0)
                ],
                dim=0,
            )
            manager_goals = goals[::self.goal_duration]
            goal_entropy = goal_entropies[::self.goal_duration]
            goal_log_prob = goal_log_probs[::self.goal_duration]
            discount_dist = self.DiscountModel(imag_modelstates)
            discount_arr = self.discount*torch.round(discount_dist.base_dist.probs)  #mean = prob(disc==1)

        lambda_returns = compute_returns(
            manager_reward[:-1],
            manager_value[:-1],
            discount_arr[::self.goal_duration][:-1],
            bootstrap=manager_value[-1],
            lambda_=self.lambda_,
        )
        manager_actor_loss, _, _ = self.manager.actor_loss(
            lambda_returns,
            discount_arr[::self.goal_duration],
            goal_entropy,
            manager_value,
            goal_log_prob,
        )
        manager_value_loss = self.manager.extrinsic_critic.loss(
            manager_modelstates,
            discount_arr[::self.goal_duration],
            lambda_returns,
        )

        worker_actor_losses = []
        worker_value_losses = []
        worker_reward = max_cosine_similarity(
            imag_modelstates,
            goals,
        )
        for t in range(0, self.horizon, self.goal_duration):
            _slice = slice(t, t + self.goal_duration)
            lambda_returns = compute_returns(
                worker_reward[_slice][:-1],  # + imag_reward[_slice][:-1] * 0.1,
                worker_value[_slice][:-1],
                discount_arr[_slice][:-1],
                bootstrap=worker_value[_slice][-1],
                lambda_=self.lambda_,
            )
            actor_loss, _, _ = self.worker.actor_loss(
                lambda_returns,
                discount_arr[_slice],
                policy_entropy[_slice],
                worker_value[_slice],
                imag_log_prob[_slice],
            )
            worker_actor_losses.append(actor_loss)
            value_loss = self.worker.critic.loss(
                imag_modelstates[_slice],
                discount_arr[_slice],
                lambda_returns,
            )
            worker_value_losses.append(value_loss)

        return (
            sum(worker_actor_losses),
            sum(worker_value_losses),
            manager_actor_loss,
            manager_value_loss,
            worker_reward,
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
            1 * (config.rssm_info["deter_size"] + config.rssm_info["stoch_size"]),
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
