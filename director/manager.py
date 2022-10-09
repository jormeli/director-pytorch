import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F

from director.utils import VectorOfCategoricals


class Manager(nn.Module):
    def __init__(
        self,
        input_size: int,
        n_latents: int,
        n_classes: int,
        n_layers: int,
        layer_size: int,
    ):
        super().__init__()

        self.input_size = input_size
        self.n_latents = n_latents
        self.n_classes = n_classes

        self.actor = nn.Sequential(
            nn.Linear(input_size, layer_size),
            nn.ELU(),
            *[
                nn.Sequential(
                    nn.Linear(layer_size, layer_size),
                    nn.ELU(),
                )
                for _ in range(1, n_layers)
            ],
            nn.Linear(layer_size, n_latents * n_classes),
            VectorOfCategoricals(n_latents, n_classes),
        )

    def forward(self, model_state: torch.Tensor) -> torch.Tensor:
        actions, logits = self.actor(model_state)
        return actions


def main():
    m = Manager(128, 8, 8, 3, 256)
    batch = torch.randn(32, 128)
    actions = m(batch)
    import pdb;pdb.set_trace()
