from typing import Tuple, List

import hydra
import torch
from einops import einops
from torch import nn, Tensor

from musep.cfg import ModelConfig, EncoderConfig, DecoderConfig, BottleneckConfig
from musep.models import STFTModule, ISTFTModule


class MusicSourceSeparationModel(nn.Module):
    """MusicSourceSeparationModel that can be used for training"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.stft = STFTModule(config.stft)
        self.istft = ISTFTModule(config.stft)
        self.complex_to_real = config.complex_to_real
        self.channels = config.channels
        self.mask = config.mask
        self.encoder = Encoder(config.encoder)
        if config.bottleneck:
            if config.bottleneck.dual_path:
                self.bottleneck = DualPathBottleneck(config.bottleneck)
            else:
                self.bottleneck = Bottleneck(config.bottleneck)
        else:
            self.bottleneck = None
        self.decoder = Decoder(config.decoder)

    def complex_as_channels(self, z: Tensor) -> Tensor:
        """
        Convert complex spectrogram to real with complex values as channels
        Args:
            z: Complex spectrogram with shape [batch, channel, freq_bins, time_frames]
        Returns:
            x: Real spectrogram with shape [batch, channel x 2 , freq_bins, time_frames]
        """

        x = torch.view_as_real(z)
        x = einops.rearrange(x, 'b c f t r -> b (c r) f t')
        return x

    def channels_as_complex(self, x: Tensor) -> Tensor:
        """
        Convert channels to complex spectrogram.
        Args:
            x: Real spectrogram with shape [batch,  channel x sources x 2 , freq_bins, time_frames]
        Returns:
            z: Complex spectrogram with shape [batch, sources, channel, freq_bins, time_frames]
        """

        x = einops.rearrange(x, 'b (s c r) f t -> b s c f t r', r=2, c=self.channels)
        z = torch.view_as_complex(x.contiguous())
        return z

    def reconstruct_phase(self, out: Tensor, z_in: Tensor) -> Tensor:
        """ Reconstruct phase from input spectrogram and apply it to the output magnitude """
        eps = 1e-8
        z_expanded = z_in.unsqueeze(1)
        if self.mask:
            z_out = out * z_expanded
        else:
            z_out = out * (z_expanded + eps / (torch.abs(z_expanded) + eps))

        return z_out

    def forward(self, input: Tensor, spectrogram_mode: bool = False) -> Tensor:
        """
        Forward pass of the model
        Args:
            input: Input tensor (waveform or spectrogram depending on config)
            spectrogram_mode: Whether the input is a waveform (True) or spectrogram (False)
        Returns: Output tensor (separated sources in waveform or spectrogram depending on config)
        """
        if spectrogram_mode:
            if input.dim() != 4:
                raise ValueError(f"Expected 4D input, got {input.dim()}D")
        else:
            if input.dim() != 3:
                raise ValueError(f"Expected 3D input, got {input.dim()}D")

        z_in = self.stft(input) if not spectrogram_mode else input

        if self.complex_to_real == 'as_channels':
            x = self.complex_as_channels(z_in)
        elif self.complex_to_real == 'as_magnitude':
            x = z_in.abs()
        else:
            raise ValueError(f"Expected complex_to_real = 'as_magnitude' or 'as_channels', got {self.complex_to_real}")

        x, res = self.encoder(x)
        if self.bottleneck:
            x = self.bottleneck(x)
        x = self.decoder(x, res)

        if self.complex_to_real == 'as_channels':
            x = self.channels_as_complex(x)
        elif self.complex_to_real == 'as_magnitude':
            x = self.reconstruct_phase(x, z_in)
        else:
            raise ValueError(f"Expected complex_to_real = 'as_magnitude' or 'as_channels', got {self.complex_to_real}")

        if not spectrogram_mode:
            x = self.istft(x)

        return x


class Encoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.encoding_blocks = nn.ModuleDict()

        for block_name, cfg in config.blocks.items():
            encoding_block = hydra.utils.instantiate(cfg)
            self.encoding_blocks[block_name] = encoding_block

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        residuals = []

        for block_name, encoding_block in  self.encoding_blocks.items():
            x = encoding_block(x)
            if isinstance(x, tuple):
                x, res = x
                residuals.append(res)

        return x, residuals


class Decoder(nn.Module):

    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.decoding_blocks = nn.ModuleDict()

        for block_name, cfg in config.blocks.items():
            decoding_block = hydra.utils.instantiate(cfg)
            self.decoding_blocks[block_name] = decoding_block

    def forward(self, x: Tensor, residuals: List[Tensor]) -> Tensor:
        for block_name, decoding_block in  self.decoding_blocks.items():
            if residuals:
                res = residuals.pop()
                x = decoding_block(x, res)
            else:
                x = decoding_block(x)

        return x


class Bottleneck(nn.Module):

    def __init__(self, config: BottleneckConfig):
        super().__init__()
        self.blocks = nn.ModuleDict()
        for block_name, cfg in config.blocks.items():
            block = hydra.utils.instantiate(cfg)
            self.blocks[block_name] = block


    def forward(self, x: Tensor) -> Tensor:

        for block_name, block in self.blocks.items():
            x = block(x)

        return x


class DualPathBottleneck(nn.Module):

    def __init__(self, config: BottleneckConfig):
        super().__init__()
        self.dual_path_first = config.dual_path_first
        self.dual_path_times = config.dual_path_times
        self.paths = nn.ModuleList()
        for i in range(self.dual_path_times * 2):
            path = nn.ModuleDict()
            for block_name, cfg in config.blocks.items():
                block = hydra.utils.instantiate(cfg)
                path[block_name] = block

            self.paths.append(path)

    def forward(self, x: Tensor) -> Tensor:
        batch = x.shape[0]
        if self.dual_path_first == 'time':
            x = einops.rearrange(x, 'b c f t -> (b f) t c')
            is_time = True
        else:
            x = einops.rearrange(x, 'b c f t -> (b t) f c')
            is_time = False

        for n, path in enumerate(self.paths):

            for block_name, block in self.paths.items():
                x = block(x)

            if n < len(self.paths) - 1:
                x = einops.rearrange(x, '(b u) v c -> (b v) u c', b=batch)  # u,v = f,t if is_time else t,f
                is_time = not (is_time)

        if is_time:
            x = einops.rearrange(x, '(b f) t c -> b c f t', b=batch)
        else:
            x = einops.rearrange(x, '(b t) f c -> b c f t', b=batch)

        return x
