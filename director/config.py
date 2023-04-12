from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from typing import Optional, Tuple
from enum import Enum


class Device(str, Enum):
    CUDA = "cuda"
    CPU = "cpu"


@dataclass
class MLPConfig:
    n_layers: int
    layer_size: int
    input_size: Optional[int] = None
    output_size: Optional[int] = None


@dataclass
class GoalAutoencoderConfig:
    n_latents: int
    n_classes: int
    encoder_cfg: MLPConfig
    decoder_cfg: MLPConfig


@dataclass
class WorkerConfig:
    action_noise: bool
    slow_target_mix: float
    extrinsic_reward: bool


@dataclass
class ManagerConfig:
    slow_target_mix: float
    intrinsic_reward: bool


@dataclass
class TrainingConfig:
    sequence_length: int
    batch_size: int
    train_every: int
    train_steps: int
    horizon: int
    seed_steps: int
    slow_target_update: int
    goal_duration: int
    save_every: int


@dataclass
class EnvironmentConfig:
    name: str
    num_parallel_envs: int
    env_args: Optional[dict]
    resize_obs: Optional[Tuple[int]] = None
    observation_shape: Optional[Tuple[int]] = None


@dataclass
class ExperimentConfig:
    training_cfg: TrainingConfig
    goal_vae_cfg: GoalAutoencoderConfig
    worker_cfg: WorkerConfig
    manager_cfg: ManagerConfig
    environment_cfg: EnvironmentConfig
    device: Device
    seed: Optional[int] = 609


ConfigStore.instance().store(name="experiment_config", node=ExperimentConfig)
