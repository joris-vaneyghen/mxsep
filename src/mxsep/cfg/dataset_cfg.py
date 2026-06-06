from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional


class MixingStrategy(Enum):
    """
    Defines the strategy for creating random mixes of stems for one e
    poch.
    """

    fixed_number = "fixed_number"
    """
    Create exactly a fixed number of examples regardless of stems exhaustion.
    """

    stop_when_any_stem_exhausted = "stop_when_any_stem_exhausted"
    """
    Create mixes until any one of the stems is exhausted, then stop.
    """

    exhaust_all_and_recycle_exhausted = "exhaust_all_and_recycle_exhausted"
    """
    Continue creating mixes until all stems are exhausted.
    Once a stem is exhausted, recycle it (reuse from beginning) and continue.
    This allows infinite mixing with all stems always available.
    """

    exhaust_all_and_drop_exhausted = "exhaust_all_and_drop_exhausted"
    """
    Continue creating mixes until all stems are exhausted.
    Once a stem is exhausted, drop it and no longer include it in subsequent mixes.
    Mixes continue with the remaining non-exhausted stems until all are exhausted.
    """


@dataclass
class AugmentationConfig:
    """
    Configuration for a single audio augmentation transformation.

    Augmentations can be applied to either individual stem before mixing,
    or to the final mixed audio.

    Example:
        >>> config = AugmentationConfig(
        ...     transformer='TimeStretch',
        ...     settings={"leave_length_unchanged": False},
        ...     p=0.5,
        ...     randomize={"min_rate": 0.8, "max_rate": 1.25},
        ...     apply_only_to=["vocals", "drums"]
        ... )
    """

    transformer: str
    """
    Class name of the augmentation transform (e.g., "TimeStretch", "PitchShift").
    Is a either a Transformer from audiomentations or a custom class implementing the same interface and registered in 
    the TransformRegistry. One can use class annotation @TransformRegistry.register() 
    """

    settings: dict[str, Any] = field(default_factory=dict)
    """
    Fixed initialization parameters for the augmentation.
    These are applied deterministically and not randomized.

    Examples:
        - For TimeStretch: {"leave_length_unchanged": False}
        - For PitchShift: {"n_steps": 2}
    """

    p: float = 0.5
    """Probability of applying this augmentation. Value between 0 and 1."""

    randomize: dict[str, Any] = field(default_factory=dict)
    """
    Parameters for randomizing the augmentation behavior.
    These define ranges or distributions for random sampling.

    Examples:
        - For Gain: {"min_gain_db": -6.0, "max_gain_db": 6.0}
        - For TimeStretch: {"min_rate": 0.8, "max_rate": 1.25}
        - For PitchShift: {"min_semitones": -2, "max_semitones": 2}
    """

    apply_only_to: Optional[list[str]] = None
    """
    List of stem names to which this augmentation should be applied.
    If None or empty, applies to all stems.

    Example: ["vocals", "drums", "bass"]
    """


@dataclass
class RandomMixConfig:
    stem_selection: Optional[dict[str, list[float]]] = None
    """ Stem selection probabilities. If not specified probabilities are calculated based on the active stems in the songs in the train dataset"""

    min_segment_length: int = 44100 * 2  # 2 seconds
    """Minimum segment length selected from start or end of audio file. (It will be padded with zero's to segment_length if it's shorter)"""

    skip_silence: str = "post_removal_of_silent_segements"  # todo implement
    """ Skip silence segments. Options are 'split_on_silence_before_segmenting', 'post_removal_of_silent_segements' """

    silence_threshold_dbfs: int = 60
    """ threshold in dBFS below which audio is considered silent (used if skip_silence is set) """

    mix_strategy: MixingStrategy = MixingStrategy.exhaust_all_and_recycle_exhausted
    """ Strategy for creating random mixes of stems for one epoch. """

    fixed_number_of_mixes: int = 0
    """ Use when mix_strategy = fixed_number """

    stem_augmentations: list[AugmentationConfig] = field(default_factory=list)
    """
    Augmentations to apply to individual stems BEFORE mixing.
    Each stem will be independently augmented according to these configurations.
    """

    mix_augmentations: list[AugmentationConfig] = field(default_factory=list)
    """
    Augmentations to apply to the final mixture AFTER mixing.
    These affect the combined audio signal.
    """


@dataclass
class AudioFilesConfig:
    audio_files_pattern: list[str] = field(default_factory=list)
    """List of Glob-like patterns with optional {stem} placeholder e.g., '/path/to/{stem}/*.mp3', '/path/**/{stem}_*.wav'"""

    audio_file_csv: list[Path] = field(default_factory=list)
    """List of csv files with columns 'stem' and 'path' for audio files"""

    predefined_jsonl_path: Optional[Path] = field(default_factory=Path)
    """ path to jsonl file or directory containing 0.jsonl ... n.jsonl (for n epochs . containing segments & mix to train or validate"""


@dataclass
class DatasetConfig:
    train:Optional[AudioFilesConfig] = None
    validation:Optional[AudioFilesConfig] = None
    test:Optional[AudioFilesConfig] = None

    sample_rate: int = 44100
    channels: int = 2
    segment_length: int = 44100 * 5  # 5 seconds
    target_sources: dict[str, list[str]] = field(default_factory=dict) # maps target_sources to stems
    random_mix: Optional[RandomMixConfig] = None