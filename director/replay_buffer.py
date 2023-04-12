import ray.rllib.utils.replay_buffers.replay_buffer as rb
from ray.rllib.policy.sample_batch import SampleBatch
import numpy as np
from collections import defaultdict
import random


class ReplayBuffer(rb.ReplayBuffer):
    def __init__(
        self,
        capacity: int,
    ):
        super().__init__(
            capacity,
            rb.StorageUnit.EPISODES,
        )

        self.ongoing_episodes = defaultdict(
            lambda: defaultdict(list)
        )
        self._len = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        worker: int,
    ):
        episode = self.ongoing_episodes[worker]
        episode[SampleBatch.OBS].append(obs)
        episode[SampleBatch.REWARDS].append(reward)
        episode[SampleBatch.DONES].append(done)
        episode[SampleBatch.ACTIONS].append(action)

        if done:
            episode_length = len(episode[SampleBatch.OBS])
            if episode_length >= 10:
                episode_id = len(self)
                super().add(
                    SampleBatch(
                        {
                            **episode,
                            SampleBatch.T: list(range(episode_length)),
                            SampleBatch.EPS_ID: [episode_id] * episode_length,
                        }
                    )
                )
                self._len += episode_length
            episode.clear()

    def sample_seq(
        self,
        seq_len: int,
    ):
        sequence = super().sample(1)
        while len(sequence) <= seq_len + 1:
            sequence = sequence.concat(super().sample(1))

        slice_start = random.randint(0, len(sequence) - seq_len - 1)
        sequence = sequence.slice(slice_start, slice_start + seq_len + 1)

        sequence[SampleBatch.OBS] = sequence[SampleBatch.OBS][1:]
        sequence[SampleBatch.REWARDS] = sequence[SampleBatch.REWARDS][:-1]
        sequence[SampleBatch.DONES] = sequence[SampleBatch.DONES][:-1]
        sequence[SampleBatch.ACTIONS] = sequence[SampleBatch.ACTIONS][:-1]
        return sequence

    def sample_batch(
        self,
        batch_size: int,
        seq_len: int,
    ):
        sequences = [self.sample_seq(seq_len) for _ in range(batch_size)]
        return {
            SampleBatch.OBS: np.stack(
                [
                    sequence[SampleBatch.OBS]
                    for sequence in sequences
                ],
                axis=1,
            ),
            SampleBatch.REWARDS: np.stack(
                [
                    sequence[SampleBatch.REWARDS]
                    for sequence in sequences
                ],
                axis=1,
            ),
            SampleBatch.DONES: np.stack(
                [
                    sequence[SampleBatch.DONES]
                    for sequence in sequences
                ],
                axis=1,
            ),
            SampleBatch.ACTIONS: np.stack(
                [
                    sequence[SampleBatch.ACTIONS]
                    for sequence in sequences
                ],
                axis=1,
            ),
        }
