from dataclasses import dataclass, field

from mxsep.cfg import DatasetConfig, ModelConfig, TrainingConfig


@dataclass
class Config:
    """Main configuration container"""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
