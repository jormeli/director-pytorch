from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from typing import Optional
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


@dataclass
class ManagerConfig:
    slow_target_mix: float


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
class ExperimentConfig:
    training_cfg: TrainingConfig
    goal_vae_cfg: GoalAutoencoderConfig
    worker_cfg: WorkerConfig
    manager_cfg: ManagerConfig
    environment: str
    device: Device
    seed: Optional[int] = 609


ConfigStore.instance().store(name="experiment_config", node=ExperimentConfig)
