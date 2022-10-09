import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F


class VectorOfCategoricals(nn.Module):
    def __init__(
        self,
        n_latents: int,
        n_classes: int,
    ):
        super().__init__()
        self.n_latents = n_latents
        self.n_classes = n_classes

    def get_dist(self, logits: torch.Tensor) -> td.Distribution:
        return td.Independent(td.OneHotCategoricalStraightThrough(logits=logits), 1)

    def forward(self, logits):
        input_shape = logits.shape
        logits = logits.view(
            *input_shape[:-1],
            self.n_latents,
            self.n_classes,
        )
        one_hot_dist = self.get_dist(logits)
        sample = one_hot_dist.sample()
        probs = F.softmax(logits, dim=-1)
        sample = sample + probs - probs.detach()
        sample = sample.view(
            *input_shape[:-1],
            self.n_latents * self.n_classes,
        )

        return sample, logits
