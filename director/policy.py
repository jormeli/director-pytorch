import torch
import torch.nn as nn
from typing import Optional

from director.utils import VectorOfCategoricals, Normal, OneHot, Dense


class Critic(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        n_layers: int,
        layer_size: int,
        slow_target_mix: Optional[float] = 1.0,
    ):
        super().__init__()
        self.slow_target_mix = slow_target_mix

        self.net = nn.Sequential(
            Dense(input_size, output_size, layer_size, n_layers),
            Normal(),
        )

        self.target = nn.Sequential(
            Dense(input_size, output_size, layer_size, n_layers),
            Normal(),
        )
        self.target.load_state_dict(self.net.state_dict())
        for param in self.target.parameters():
            param.requires_grad = False

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def target_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.target(x)

    def loss(
        self,
        imag_modelstates: torch.Tensor,
        discount_arr: torch.Tensor,
        lambda_returns: torch.Tensor,
    ) -> torch.Tensor:
        discount_arr = torch.cat([torch.ones_like(discount_arr[:1]), discount_arr[1:]])
        discount = torch.cumprod(discount_arr[:-1], 0)
        with torch.no_grad():
            value_modelstates = imag_modelstates[:-1].detach()
            value_discount = discount.detach()
            value_target = lambda_returns.detach()

        value_dist = self.value(value_modelstates)
        value_loss = -torch.mean(value_discount * value_dist.log_prob(value_target).unsqueeze(-1))
        return value_loss

    def update_target(self):
        mix = self.slow_target_mix
        for param, target_param in zip(
            self.net.parameters(), self.target.parameters()
        ):
            target_param.data.copy_(mix * param.data + (1 - mix) * target_param.data)


class Manager(nn.Module):
    def __init__(
        self,
        input_size: int,
        n_latents: int,
        n_classes: int,
        n_layers: int,
        layer_size: int,
        slow_target_mix: float,
    ):
        super().__init__()

        self.input_size = input_size
        self.n_latents = n_latents
        self.n_classes = n_classes
        self.lambda_ = 0.95
        self.actor_entropy_scale = 1e-3
        self.grad = "reinforce"

        self.actor = nn.Sequential(
            Dense(input_size, n_latents * n_classes, layer_size, n_layers),
            VectorOfCategoricals(n_latents, n_classes),
        )

        self.extrinsic_critic = Critic(
            input_size=input_size,
            output_size=1,
            n_layers=n_layers,
            layer_size=layer_size,
            slow_target_mix=slow_target_mix,
        )
        self.intrinsic_critic = Critic(
            input_size=input_size,
            output_size=1,
            n_layers=n_layers,
            layer_size=layer_size,
            slow_target_mix=slow_target_mix,
        )

    def get_action(self, model_state: torch.Tensor) -> torch.Tensor:
        return self.actor(model_state)

    def get_value(self, model_state: torch.Tensor) -> torch.Tensor:
        return self.critic(model_state).rsample()

    def actor_loss(
        self,
        ext_lambda_returns: torch.Tensor,
        int_lambda_returns: torch.Tensor,
        discount_arr: torch.Tensor,
        policy_entropy: torch.Tensor,
        ext_value: torch.Tensor,
        int_value: torch.Tensor,
        imag_log_prob: torch.Tensor,
    ) -> torch.Tensor:
        ext_advantage = (ext_lambda_returns - ext_value[:-1]).detach()
        int_advantage = (int_lambda_returns - int_value[:-1]).detach()
        advantage = ext_advantage + int_advantage * 0.1
        objective = imag_log_prob[1:].unsqueeze(-1) * advantage

        discount_arr = torch.cat([torch.ones_like(discount_arr[:1]), discount_arr[1:]])
        discount = torch.cumprod(discount_arr[:-1], 0)
        policy_entropy = policy_entropy[1:].unsqueeze(-1)
        actor_loss = -torch.sum(torch.mean(discount * (objective + self.actor_entropy_scale * policy_entropy), dim=1))
        return actor_loss, discount

    def extrinsic_critic_loss(
        self,
        imagined_model_states: torch.Tensor,
        discount: torch.Tensor,
        lambda_returns: torch.Tensor,
    ) -> torch.Tensor:
        return self.extrinsic_critic.loss(
            imagined_model_states,
            discount,
            lambda_returns,
        )

    def intrinsic_critic_loss(
        self,
        imagined_model_states: torch.Tensor,
        discount: torch.Tensor,
        lambda_returns: torch.Tensor,
    ) -> torch.Tensor:
        return self.intrinsic_critic.loss(
            imagined_model_states,
            discount,
            lambda_returns,
        )


class Worker(nn.Module):
    def __init__(
        self,
        input_size: int,
        action_size: int,
        n_layers: int,
        layer_size: int,
        slow_target_mix: float,
    ):
        super().__init__()
        self.input_size = input_size
        self.action_size = action_size
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.lambda_ = 0.95
        self.actor_entropy_scale = 1e-3
        self.grad = "reinforce"
        self.noise_amount = 0.1

        self.actor = nn.Sequential(
            Dense(input_size * 2, action_size, layer_size, n_layers),
            OneHot(action_size),
        )

        self.critic = Critic(
            input_size=input_size * 2,
            output_size=1,
            n_layers=n_layers,
            layer_size=layer_size,
            slow_target_mix=slow_target_mix,
        )

    def forward(self, x):
        return self.actor(x)

    def get_action(
        self,
        model_state: torch.Tensor,
        goal: torch.Tensor,
    ):
        return self.actor(torch.cat([model_state, goal], dim=-1))

    def value_loss(
        self,
        imagined_model_states: torch.Tensor,
        discount: torch.Tensor,
        lambda_returns: torch.Tensor,
    ) -> torch.Tensor:
        return self.critic.loss(
            imagined_model_states,
            discount,
            lambda_returns,
        )

    def actor_loss(
        self,
        lambda_returns: torch.Tensor,
        discount_arr: torch.Tensor,
        policy_entropy: torch.Tensor,
        imag_value: torch.Tensor,
        imag_log_prob: torch.Tensor,
    ) -> torch.Tensor:
        advantage = (lambda_returns - imag_value[:-1]).detach()
        objective = imag_log_prob[1:].unsqueeze(-1) * advantage

        discount_arr = torch.cat([torch.ones_like(discount_arr[:1]), discount_arr[1:]])
        discount = torch.cumprod(discount_arr[:-1], 0)
        policy_entropy = policy_entropy[1:].unsqueeze(-1)
        actor_loss = -torch.sum(torch.mean(discount * (objective + self.actor_entropy_scale * policy_entropy), dim=1))
        return actor_loss, discount, lambda_returns
