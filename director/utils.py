import torch
import torch.distributions as td
import torch.nn as nn
from typing import Optional


class Dense(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        layer_size: int,
        n_layers: int,
        distribution: Optional[str] = None,
        layer_norm: Optional[bool] = True,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, layer_size),
            nn.LayerNorm(layer_size, elementwise_affine=False)
            if layer_norm
            else nn.Identity(),
            nn.ELU(),
            *[
                nn.Sequential(
                    nn.Linear(layer_size, layer_size),
                    nn.LayerNorm(layer_size, elementwise_affine=False)
                    if layer_norm
                    else nn.Identity(),
                    nn.ELU(),
                )
                for _ in range(1, n_layers)
            ],
            nn.Linear(layer_size, output_size),
        )

    def forward(self, inputs):
        return self.net(inputs)


class VectorOfCategoricals(nn.Module):
    def __init__(
        self,
        n_latents: int,
        n_classes: int,
    ):
        super().__init__()
        self.n_latents = n_latents
        self.n_classes = n_classes

    @staticmethod
    def get_dist(
        logits: torch.Tensor,
        independent: Optional[bool] = False,
    ) -> td.Distribution:
        dist = td.OneHotCategoricalStraightThrough(logits=logits)
        if independent:
            dist = td.Independent(dist, 1)
        return dist

    def forward(self, logits):
        input_shape = logits.shape
        logits = logits.view(
            *input_shape[:-1],
            self.n_latents,
            self.n_classes,
        )
        one_hot_dist = self.get_dist(logits)
        sample = one_hot_dist.rsample()
        sample = sample.view(
            *input_shape[:-1],
            self.n_latents * self.n_classes,
        )

        return sample, one_hot_dist.logits


class OneHot(nn.Module):
    def __init__(
        self,
        n_classes: int,
    ):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, logits: torch.Tensor) -> None:
        distribution = td.OneHotCategoricalStraightThrough(logits=logits)
        return distribution


class Normal(nn.Module):
    def forward(self, inputs):
        distribution = td.Independent(td.Normal(inputs, 1), 1)
        return distribution


class EMA:
    def __init__(
        self,
        alpha=0.05,
    ):
        self.alpha = 0.05
        self._loc = None
        self._scale = None

    @property
    def loc(self):
        return self._loc

    @property
    def scale(self):
        return self._scale

    def update(self, scale, loc):
        alpha = self.alpha
        if self._loc is None or self._scale is None:
            alpha = 1.0
            self._loc, self._scale = 0, 0

        self._loc = self._loc * (1 - alpha) + loc * alpha
        self._scale = self._scale * (1 - alpha) + scale * alpha


def max_cosine_similarity(
    inputs: torch.Tensor,
    other: torch.Tensor,
    dim: Optional[int] = -1,
) -> torch.Tensor:
    """Implements max-cosine similarity between two tensors,
    as defined in the paper.
    """
    inputs_norm = torch.norm(inputs, dim=dim)
    other_norm = torch.norm(other, dim=dim)
    max_norm = torch.max(
        torch.stack([inputs_norm, other_norm], dim=0),
        dim=0,
    ).values.unsqueeze(dim)
    max_cos_sim = torch.sum((inputs / max_norm) * (other / max_norm), dim=dim)
    return max_cos_sim.unsqueeze(dim)


def compute_returns(
    reward: torch.Tensor,
    value: torch.Tensor,
    discount: torch.Tensor,
    bootstrap: torch.Tensor,
    lambda_: float,
) -> torch.Tensor:
    next_values = torch.cat([value[1:], bootstrap[None]], 0)
    target = reward + discount * next_values * (1 - lambda_)
    timesteps = list(range(reward.shape[0] - 1, -1, -1))
    outputs = []
    accumulated_reward = bootstrap
    for t in timesteps:
        inp = target[t]
        discount_factor = discount[t]
        accumulated_reward = inp + discount_factor * lambda_ * accumulated_reward
        outputs.append(accumulated_reward)
    returns = torch.cat(
        [
            torch.flip(torch.stack(outputs), [0]),
            bootstrap.unsqueeze(0),
        ],
        dim=0,
    )

    return returns
