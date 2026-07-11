from typing import Optional

import torch
from einops import einops
from torch import nn

from mxsep.cfg import STFTConfig


class STFTModule(nn.Module):

    def __init__(self, config: STFTConfig):
        super().__init__()
        self.config = config
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.win_length = config.win_length
        self.window = torch.hann_window(config.win_length) # todo if config.window == 'hann' else
        self.keep_freq_bins = config.keep_freq_bins

        # Validate that we're not trying to keep more bins than available
        total_bins = self.n_fft // 2 + 1  # For real STFT
        if self.keep_freq_bins == -1:
            self.keep_freq_bins = total_bins  # Keep all bins by default
        elif self.keep_freq_bins > total_bins:
            raise ValueError(
                f"Cannot keep {self.keep_freq_bins} frequency bins when only {total_bins} "
                f"are available from STFT (n_fft={self.n_fft})"
            )


    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
            Compute STFT
        Args:
            waveform: Input waveform with shape [batch, channels, time] or [batch, sources, channels, time]

        Returns:
            Spectrogram with shape [batch, channels, freq_bins, time_frames, real_imag] or [batch, sources, channels, freq_bins, time_frames, real_imag]

        """
        if waveform.dim() not in [3, 4]:
            raise ValueError(f"Expected 3D or 4D input, got {waveform.dim()}D")

        # Move window to correct device
        window = self.window.to(waveform.device)

        sources = waveform.shape[1] if waveform.dim() == 4 else 0
        batch = waveform.shape[0]
        channels = waveform.shape[-2]

        if sources :
            # [batch, sources, channels, time] -> [batch * sources * channels, time]
            x = einops.rearrange(waveform, 'b s c t -> (b s c) t')
        else:
            # [batch, channels, time] -> [batch * channels, time]
            x = einops.rearrange(waveform, 'b c t -> (b c) t')

        # Compute STFT
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            normalized=self.config.normalized,
            return_complex=True
        )

        # Trim frequency bins: keep only the first `keep_freq_bins` bins
        # x shape after STFT: [batch * sources * channels, total_freq_bins, time_frames]
        x = x[:, :self.keep_freq_bins, :]

        if sources :
            x = einops.rearrange(x, '(b s c) f t-> b s c f t',
                                 b=batch,
                                 s=sources,
                                 c=channels)
        else:
            x = einops.rearrange(x, '(b c) f t -> b c f t',
                                 b=batch,
                                 c=channels)

        x = torch.view_as_real(x)
        return x



class ISTFTModule(nn.Module):

    def __init__(self, config: STFTConfig):
        super().__init__()
        self.config = config
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.win_length = config.win_length
        self.window = torch.hann_window(config.win_length)
        self.keep_freq_bins = config.keep_freq_bins

        # Validate that we're not trying to keep more bins than available
        total_bins = self.n_fft // 2 + 1  # For real STFT
        if self.keep_freq_bins == -1:
            self.keep_freq_bins = total_bins  # Keep all bins by default
        elif self.keep_freq_bins > total_bins:
            raise ValueError(
                f"Cannot keep {self.keep_freq_bins} frequency bins when only {total_bins} "
                f"are available from STFT (n_fft={self.n_fft})"
            )

        # Check if n_fft is power of two
        def is_power_of_two(n):
            return n > 0 and (n & (n - 1)) == 0

        self.n_fft_is_power_of_two = is_power_of_two(self.n_fft)




    def forward(self, spectrogram: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        """
        Compute inverse STFT (ISTFT)

        Args:
            spectrogram: Tensor with shape [batch, channels, freq_bins, time_frames, real_imag] or [batch, sources, channels, freq_bins, time_frames, real_imag]
            length: Optional output length for the time dimension

        Returns:
            Waveform with shape [batch, channels, time] or [batch, sources, channels, time]
        """
        if spectrogram.dim() not in [5, 6]:
            raise ValueError(f"Expected 4D or 5D input, got {spectrogram.dim()}D")
        
        if spectrogram.dtype == torch.bfloat16:
            spectrogram = spectrogram.to(dtype=torch.float32)
        spectrogram = torch.view_as_complex(spectrogram.contiguous())

        if spectrogram.dim() == 5:
            batch, sources, channels, freq_bins, time_frames = spectrogram.shape
        else:
            sources = 0
            batch, channels, freq_bins, time_frames = spectrogram.shape

        # Move window to correct device
        window = self.window.to(spectrogram.device)

        # Get original total number of frequency bins for reconstruction
        total_bins = self.n_fft // 2 + 1

        # Create padded spectrogram with zeros for discarded high frequencies
        target_shape = spectrogram.shape[:-2] + (total_bins, time_frames)  # [batch, (sources,) channels, total_freq_bins, time_frames]
        padded_spectrogram = torch.zeros(size=target_shape, dtype=spectrogram.dtype,device=spectrogram.device)
        padded_spectrogram[..., :self.keep_freq_bins, :] = spectrogram

        # Rearrange for ISTFT
        if sources:
            x = einops.rearrange(padded_spectrogram, 'b s c f t -> (b s c) f t')
        else:
            x = einops.rearrange(padded_spectrogram, 'b c f t -> (b c) f t')

        # cuFFT only supports dimensions whose sizes are powers of two when computing in half precision
        if x.dtype == torch.complex32 and not self.n_fft_is_power_of_two:
            # convert to full precision
            x = x.to(torch.complex64)

        # Compute ISTFT
        x = torch.istft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            normalized=self.config.normalized,
            length=length,
            return_complex=False
        )


        # Reshape back to original format
        if sources:
            x = einops.rearrange(x, '(b s c) t -> b s c t',
                                 b=batch,
                                 s=sources,
                                 c=channels)
        else:
            x = einops.rearrange(x, '(b c) t -> b c t',
                                 b=batch,
                                 c=channels)

        return x