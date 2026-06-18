from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple

import yaml

from mxsep.utils.serialize import asdict_filter_empty


class PredefinedType(Enum):
    """
    Defines the type of predefined mixing examples to use during training.

    Predefined mixing examples are usually dynamically created training examples by mixing
    different audio sources (e.g., vocals, drums, bass) with random gains and other augmentations

    """


    PREDEFINED_MIXES = "predefined_mixes"
    """
    We use predefined mix examples defined by a jsonl file in the setting 'predefined_mix_path'.
    Each line in the jsonl file specifies a mixing configuration (stem, audio segment, augmentation).
    The same shuffled mixing examples are reused for the entire training run. 
    """

    PREDEFINED_MIXES_PER_EPOCH = "predefined_mixes_per_epoch"
    """
    Per epoch we use other predefined mix examples. The setting 'predefined_mixes_path' is a directory with files: 0.jsonl ... n.jsonl.
    Each file contains examples of the same mixing strategy. We don't shuffle the examples
    This allows for different mixing strategies across epochs. 
    """


@dataclass
class DatasetTrainingConfig:
    """
    Configuration for dataset loading and preprocessing during training.

    This class handles how audio data is loaded, mixed, and prepared for the model.
    It supports both dynamic random mixing and predefined mixing strategies.

    Attributes:
        random_mix_config_path: Path to YAML config file specifying random mixing parameters
            (e.g., gain ranges, source probabilities, target loudness).
        predefined_path: Path to either:
            - A single .jsonl file
            - A directory containing 0.jsonl, 1.jsonl, ...
        predefined_type: Whether to use 'predefined_mixes' (single jsonl) or 'predefined_mixes_per_epoch' (directory with epoch files)
        target_source_stem_mapping: Mapping from target stems (e.g., "vocals") to source stems
            in the dataset. Example: {"vocals": ["lead_vocal", "backing_vocal"]}
    """

    random_mix_config_path: str
    """Path to YAML config file with datasets and random mixing parameters (e.g., augmentations, source probabilities)"""

    predefined_path: str
    """Path to predefined mix file(s) - either a single .jsonl or directory with epoch files"""

    predefined_type: PredefinedType
    """Type of predefined examples to use - either 'predefined_mixes' or 'predefined_mixes_per_epoch'"""

    target_source_stem_mapping: Dict[str, List[str]] = field(default_factory=dict)
    """Maps model target sources to dataset source stems (e.g., {'wind_instr': ['trumpet', 'flute', 'saxophone']})"""

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return asdict_filter_empty(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetTrainingConfig':
        """
        Create DatasetTrainingConfig from dictionary.

        Args:
            data: Dictionary containing configuration parameters

        Returns:
            DatasetTrainingConfig instance

        Raises:
            ValueError: If random_mix_type is invalid or required fields are missing
        """
        # Create a copy to avoid mutating the input
        data_copy = data.copy()

        # Convert random_mix_type from string to Enum if needed
        if 'random_mix_type' in data_copy:
            if isinstance(data_copy['random_mix_type'], str):
                try:
                    data_copy['random_mix_type'] = PredefinedType(data_copy['random_mix_type'])
                except ValueError as e:
                    valid_values = [m.value for m in PredefinedType]
                    raise ValueError(
                        f"Invalid random_mix_type: {data_copy['random_mix_type']}. "
                        f"Must be one of: {valid_values}"
                    ) from e
        #
        # # Validate required fields
        # required_fields = ['random_mix_config_path', 'predefined_mix_path', 'random_mix_type']
        # for field_name in required_fields:
        #     if field_name not in data_copy:
        #         raise ValueError(f"Missing required field: {field_name}")

        return cls(**data_copy)


@dataclass
class TrainingConfig:
    """
    Main training configuration for the mxsep model.

    This class consolidates all training-related configuration including dataset,
    model, optimization, and evaluation settings.

    Attributes:
        dataset: DatasetTrainingConfig instance for data loading and preprocessing
        model_config_path: Path to YAML file containing model architecture configuration
        stft_device: Device for STFT computation ('cpu', 'cuda', 'cuda:0', etc.)
        scheduler: scheduler name. Valid options: 'LRScheduler'  or full class path for custom scheduler
        scheduler_params: parameters for learning rate scheduling
        optimizer: optimizer name, valid options: 'AdamW', 'SGD' or full class path for custom optimizer
        optimizer_param: parameters for model optimization
        wandb_key: Weights & Biases API key for experiment tracking (optional)
        loss_function: Loss function to use ('l1', 'l2', 'multi_resolution_stft', etc.)
        evaluation_metric: List of metrics to compute during evaluation (e.g., ['sisdr', 'pesq'])
    """

    train_set: DatasetTrainingConfig
    """Dataset configuration for training"""

    validation_set: DatasetTrainingConfig
    """Dataset configuration for validation"""

    model_config_path: str
    """Path to YAML file with model architecture parameters"""

    stft_device: str
    """Device for STFT computation - use 'cpu' for memory efficiency or 'cuda' for speed"""

    scheduler:str
    """Learning rate scheduler identifier - options: 'cosine', 'step', 'plateau', or full class path for custom scheduler"""

    scheduler_params:Dict[str, Any]
    """Parameters for learning rate scheduling"""

    optimizer: str
    """Optimizer identifier - options: 'adam', 'sgd', or full class path for custom optimizer"""

    optimizer_params: Dict[str, Any]
    """Parameters for model optimization"""

    wandb_key: str
    """Weights & Biases API key for logging (set empty string to disable)"""

    loss_function: str
    """Loss function identifier - options: 'l1', 'l2', 'multi_resolution_stft', 'hybrid'"""

    evaluation_metric: List[str]
    """List of evaluation metrics - e.g., ['si_sdr', 'pesq', 'stoi']"""

    mixed_precision: bool = False
    """Enable automatic mixed precision training"""

    gradient_clipping: Optional[float] = None
    """Max gradient norm for clipping"""

    sharding: bool = False
    """Enable model sharding for large models"""

    data_parallel: bool = False
    """Enable DataParallel for multi-GPU"""

    distributed_training: bool = False
    """Enable DistributedDataParallel"""

    activation_checkpointing: bool = False
    """Trade compute for memory"""

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary for serialization.

        Returns:
            Dictionary representation with nested objects properly serialized
        """
        return asdict_filter_empty(self)


    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingConfig':
        """
        Create TrainingConfig from dictionary.

        Args:
            data: Dictionary containing configuration parameters

        Returns:
            TrainingConfig instance

        Raises:
            ValueError: If required fields are missing or have invalid values
        """
        # Create a copy to avoid mutating the input
        data_copy = data.copy()

        # Handle nested dataset configuration
        if 'train_set' in data_copy:
            if isinstance(data_copy['train_set'], dict):
                data_copy['train_set'] = DatasetTrainingConfig.from_dict(data_copy['train_set'])
            elif not isinstance(data_copy['train_set'], DatasetTrainingConfig):
                raise ValueError(
                    f"dataset must be dict or DatasetTrainingConfig, got {type(data_copy['train_set'])}"
                )
        if 'validation_set' in data_copy:
            if isinstance(data_copy['validation_set'], dict):
                data_copy['validation_set'] = DatasetTrainingConfig.from_dict(data_copy['validation_set'])
            elif not isinstance(data_copy['validation_set'], DatasetTrainingConfig):
                raise ValueError(
                    f"dataset must be dict or DatasetTrainingConfig, got {type(data_copy['validation_set'])}"
                )

        # # Validate required fields
        # required_fields = [
        #     'dataset', 'model_config_path', 'stft_device',
        #     'scheduler', 'optimizer', 'loss_function', 'evaluation_metric'
        # ]
        # for field_name in required_fields:
        #     if field_name not in data_copy:
        #         raise ValueError(f"Missing required field: {field_name}")
        #
        # # Validate field types
        # if not isinstance(data_copy['evaluation_metric'], list):
        #     raise ValueError("evaluation_metric must be a list of strings")

        return cls(**data_copy)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'TrainingConfig':
        """
        Load configuration from a YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            TrainingConfig instance

        Raises:
            FileNotFoundError: If the YAML file doesn't exist
            yaml.YAMLError: If the YAML is malformed
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """
        Save configuration to a YAML file.

        Args:
            path: Path where to save the YAML file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, sort_keys=False, indent=2)


if __name__ == '__main__':
    # Example usage
    config = TrainingConfig(
        train_set=DatasetTrainingConfig(
            random_mix_config_path="configs/random_mix.yaml",
            predefined_path="configs/predefined_mixes.jsonl",
            predefined_type=PredefinedType.PREDEFINED_MIXES,
            target_source_stem_mapping={"vocals": ["lead_vocal", "backing_vocal"]}
        ),
        validation_set=DatasetTrainingConfig(
            random_mix_config_path="configs/random_mix.yaml",
            predefined_path="configs/predefined_mixes.jsonl",
            predefined_type=PredefinedType.PREDEFINED_MIXES,
            target_source_stem_mapping={"vocals": ["lead_vocal", "backing_vocal"]}
        ),
        model_config_path="configs/model.yaml",
        stft_device="cuda",
        scheduler="cosine",
        scheduler_params={"T_max": 100},
        optimizer="adam",
        optimizer_params={"lr": 1e-4},
        wandb_key="your_wandb_key_here",
        loss_function="multi_resolution_stft",
        evaluation_metric=["si_sdr", "pesq"]
    )

    # Save to YAML
    config.to_yaml("training_config.yaml")

    # # Load from YAML
    loaded_config = TrainingConfig.from_yaml("training_config.yaml")
    print(loaded_config)


@dataclass
class WandbConfig:
    api_key: str = "my_secret"
    project: str = "Music Source Separation"
    job_type: str = "training"
    name: str = "experiment_name"
    tags: list = field(default_factory=list)
    notes: str = ''
    run_id: Optional[str] = None
    resume: str = "allow"
    fork_from: Optional[str] = None # should be {run_id}?_step={step}
    watch:Optional[dict] = None


@dataclass
class MonitoringConfig:
    wandb: Optional[WandbConfig] = None
    log_interval: int = 1
    show_progress_bar: bool = True
    save_interval: int = 1000
    debug_memory: bool = False


class MusSepLossDomain(Enum):
    time_domain = 'time_domain'
    tf_domain = 'tf_domain'


@dataclass
class TrainingConfig:
    seed:int = 42
    deterministic:bool= True
    optimizer: dict = field(default_factory=dict)
    lr_scheduler: Optional[dict] = None
    loss: dict = field(default_factory=dict)
    mus_sep_loss_domain: MusSepLossDomain = MusSepLossDomain.time_domain
    epochs: int = 100
    use_amp: bool = True
    gradient_clip: Optional[float] = None
    batch_size: int = 8
    checkpoint_interval: int = 1
    checkpoint_dir: Path = field(default_factory=Path)
    resume_from_checkpoint: Optional[Path] = None
    num_workers: int = 4
    stft_device: str = 'cuda'
    device: str = 'cuda'
    device_ids: list = field(default_factory=list)  # for distributed training, e.g. [0, 1, 2, 3]
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
