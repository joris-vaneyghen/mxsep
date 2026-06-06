from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class STFTConfig:
    """Configuration for STFT/ISTFT operations"""

    n_fft: int = 2048
    win_length: int = 2048
    hop_length: int = 512
    window: str = "hann"
    normalized: bool = False
    keep_freq_bins: int = -1


@dataclass
class EncoderConfig:
    """Configuration for Encoder operations"""

    blocks: dict[str, dict] = field(default_factory=dict)


@dataclass
class DecoderConfig:
    """Configuration for Decoder operations"""

    blocks: dict[str, dict] = field(default_factory=dict)


@dataclass
class BottleneckConfig:
    """Configuration for Bottleneck operations for example Dual RNN or Transformer blocks"""

    dual_path: bool = False
    dual_path_first: str = 'time'
    dual_path_times: int = 1

    blocks: dict[str, dict] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Configuration for Model operations"""

    name: str = field(default_factory=str)
    target_sources: list[str] = field(default_factory=list)
    dim: dict[str, int] = field(default_factory=dict)
    stft: STFTConfig = field(default_factory=STFTConfig)
    complex_to_real: str = 'as_channels'
    channels: int = 2
    segment_length: int = field(default_factory=int)
    sample_rate: int = field(default_factory=int)
    mask: str = 'none'
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    bottleneck: Optional[BottleneckConfig] = field(default=None)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
