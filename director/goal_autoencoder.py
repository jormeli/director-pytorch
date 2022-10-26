import torch
import torch.distributions as td
import torch.nn as nn
from director.utils import VectorOfCategoricals, Normal
from director.utils import Dense

from typing import Tuple, Dict


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
            Dense(input_size, n_latents * n_classes, 512, 3),
            self.categoricals,
        )

        self.decoder = nn.Sequential(
            Dense(n_latents * n_classes, input_size, 512, 3),
            Normal(),
        )

    def get_dist(
        self,
        logits: torch.Tensor,
    ) -> td.Distribution:
        self.categoricals.get_dist(logits)

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
        dist = self.decoder(samples)
        return dist.mean, dist

    def loss(
        self,
        inputs: torch.Tensor,
        reconstruction_dist: td.Distribution,
        logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        reconstruction_loss = -torch.mean(reconstruction_dist.log_prob(inputs))
        posterior = self.categoricals.get_dist(logits)
        prior = self.categoricals.get_dist(torch.ones_like(logits))
        kl_divergence = td.kl_divergence(posterior, prior).sum(dim=-1).mean()

        return {
            "loss": reconstruction_loss + kl_divergence,
            "reconstruction_loss": reconstruction_loss,
            "kl_divergence": kl_divergence,
        }
