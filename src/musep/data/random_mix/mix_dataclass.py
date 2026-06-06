import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Self

import pandas as pd
from audiomentations.core.transforms_interface import BaseWaveformTransform

from musep.cfg import AugmentationConfig
from musep.utils.serialize import asdict_filter_empty


@dataclass
class Augmentation:
    transformer: str
    """
    Class name of the augmentation transform (e.g., "TimeStretch", "PitchShift").
    Is a either a Transformer from audiomentations or a custom class implementing the same interface and registered in 
    the TransformRegistry. One can use class annotation @TransformRegistry.register() 
    """

    settings: Dict[str, Any] = field(default_factory=dict)
    """
    Fixed initialization parameters for the augmentation.
    These are applied deterministically and not randomized.

    Examples:
        - For TimeStretch: {"leave_length_unchanged": False}
        - For PitchShift: {"n_steps": 2}
    """

    parameters: Dict[str, Any] = field(default_factory=dict)
    """
    parameters will be set to transformer.parameters
    """

    def to_dict(self) -> Dict[str, Any]:
        return asdict_filter_empty(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """Create Segment from dictionary."""
        return cls(**data)

    @classmethod
    def create_from(cls, config: AugmentationConfig, transformer: BaseWaveformTransform):
        parameters = {key: value for key, value in transformer.parameters.items() if key != 'should_apply'}
        return cls(transformer=config.transformer, settings=config.settings, parameters=parameters)


@dataclass
class Segment:
    path: str
    stem: str
    offset: int
    length: int
    augmentations: List[Augmentation] = field(default_factory=list)

    def __post_init__(self):
        """Validate data after initialization."""
        if self.length <= 0:
            raise ValueError(f"Length should be positive: {self.length}")
        if self.offset < 0:
            raise ValueError(f"Offset cannot be negative: {self.offset}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict_filter_empty(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        if 'augmentations' in data:
            data['augmentations'] = [
                Augmentation.from_dict(augm_data) if isinstance(augm_data, dict) else augm_data
                for augm_data in data['augmentations']
            ]
        return cls(**data)

    def add_augmentation(self, name: str, params: Dict[str, Any]) -> None:
        """Add an augmentation to the segment."""
        self.augmentations[name] = params

    def remove_augmentation(self, name: str) -> None:
        """Remove an augmentation by name."""
        self.augmentations.pop(name, None)

    def get_augmentation(self, name: str) -> Optional[Dict[str, Any]]:
        """Get augmentation parameters by name."""
        return self.augmentations.get(name)

    def has_augmentation(self, name: str) -> bool:
        """Check if segment has a specific augmentation."""
        return name in self.augmentations

    def clear_augmentations(self) -> None:
        """Remove all augmentations."""
        self.augmentations.clear()

    @property
    def end_position(self) -> int:
        """Calculate the end position of the segment."""
        return self.position + int(self.length)


@dataclass
class Mix:
    segments: List[Segment] = field(default_factory=list)
    mix_augmentations: List[Augmentation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict_filter_empty(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """Create Mix from dictionary."""
        if 'segments' in data:
            data['segments'] = [
                Segment.from_dict(segm_data) if isinstance(segm_data, dict) else segm_data
                for segm_data in data['segments']
            ]
        if 'mix_augmentations' in data:
            data['mix_augmentations'] = [
                Augmentation.from_dict(augm_data) if isinstance(augm_data, dict) else augm_data
                for augm_data in data['mix_augmentations']
            ]
        return cls(**data)

    def add_segment(self, segment: Segment) -> None:
        """Add a segment to the mix and update its reference."""
        self.segments.append(segment)

    def remove_segment(self, segment: Segment) -> None:
        """Remove a segment from the mix."""
        if segment in self.segments:
            self.segments.remove(segment)

    def get_segments_by_stem(self, stem: str) -> List[Segment]:
        """Get all segments from a specific stem."""
        return [seg for seg in self.segments if seg.stem == stem]


