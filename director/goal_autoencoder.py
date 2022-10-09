import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from director.utils import VectorOfCategoricals

from typing import Tuple


class GoalAutoencoder(nn.Module):
    def __init__(
        self,
        input_size: int,  # latent_dim
        n_latents: int,
        n_classes: int,
    ):
        super().__init__()
        self.input_size = input_size
        self.n_latents = n_latents
        self.n_classes = n_classes

        self.categoricals = VectorOfCategoricals(n_latents, n_classes)
        self.encoder = nn.Sequential(
            nn.Linear(input_size, n_latents * n_classes),
            self.categoricals,
        )

        self.decoder = nn.Sequential(
            nn.Linear(n_latents * n_classes, input_size),
        )

    def get_dist(
        self,
        logits: torch.Tensor,
    ) -> td.Distribution:
        return td.Independent(td.OneHotCategoricalStraightThrough(logits=logits), 1)

    def encode(
        self,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sample, logits = self.encoder(batch)
        return sample, logits

    def decode(
        self,
        samples: torch.Tensor,
    ) -> torch.Tensor:
        input_shape = samples.shape
        states = self.decoder(samples)
        return states


    def loss(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        reconstruction_loss = torch.mean((inputs - reconstructions) ** 2)
        kl_div = torch.mean(
            torch.distributions.kl.kl_divergence(
                self.categoricals.get_dist(logits),
                self.categoricals.get_dist(
                    F.softmax(torch.ones_like(logits), dim=-1)
                ),
            ),
        )

        return reconstruction_loss + 1.0 * kl_div


def main():
    ae = GoalAutoencoder(128, 8, 8)
    batch = torch.randn(32, 128)
    samples, logits = ae.encode(batch)
    rec = ae.decode(samples)
    loss = ae.loss(batch, rec, logits)
    import pdb;pdb.set_trace()
