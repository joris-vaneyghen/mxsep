from typing import Tuple

import torch
from einops import einops
from torch import nn
from mxsep.models.modules.transformer import RMSNorm


class BandSplit(nn.Module):
    def __init__(
            self,
            dim_in,
            dim_out,
            nb_bands_per_split: Tuple[int, ...] = (24, 12, 8, 8, 8, 2),
            freq_bins_per_band_per_split: Tuple[int, ...] = (2, 4, 12, 24, 48, 128),
    ):
        super().__init__()
        assert len(nb_bands_per_split) == len(freq_bins_per_band_per_split)
        self.nb_bands_per_split = nb_bands_per_split
        self.freq_bins_per_band_per_split = freq_bins_per_band_per_split
        self.split_sizes = tuple(nb_bands*freq_bins for (nb_bands, freq_bins) in zip(nb_bands_per_split, freq_bins_per_band_per_split) )
        self.total_nb_bands = sum(nb_bands_per_split) 
        self.weights_per_split = nn.ParameterList([])
        self.bias = nn.Parameter(torch.randn(self.total_nb_bands, dim_out ))
        nn.init.normal_(self.bias, std=0.02)
        self.rmsnorm_per_split = nn.ModuleList([])

        for (nb_bands, freq_bins) in zip(nb_bands_per_split, freq_bins_per_band_per_split):
            self.rmsnorm_per_split.append(RMSNorm(dim_in * freq_bins)) # todo Should we use RMSNorm with learned weights per band as in Original impl?
            weights = nn.Parameter(torch.randn(nb_bands, dim_out, dim_in * freq_bins))
            nn.init.kaiming_uniform_(weights)
            self.weights_per_split.append(weights)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :type x: Tensor with shape (batch_size, dim_in, freq_bins, time_frames)
        
        :return split_output: tensor with shape (batch_size, dim_out, nb_bands, time_frames)
        """
        x = x.split(self.split_sizes, dim=-2)

        outs = []
        for x_i, w_i, rmsnorm, n_i, f_i in zip(x, self.weights_per_split, self.rmsnorm_per_split, self.nb_bands_per_split, self.freq_bins_per_band_per_split):
            x_i = einops.rearrange(x_i, "b d (n f) t -> b t n (f d)", n=n_i, f=f_i)
            x_i = rmsnorm(x_i)
            x_i = torch.einsum("btni,noi->btno", x_i, w_i)
            outs.append(x_i)


        x = torch.cat(outs, dim=-2)
        x = x  + self.bias.unsqueeze(0).unsqueeze(0)
        return einops.rearrange(x, "b t n d -> b d n t")



# if __name__ == '__main__':
#     dim_in = 4
#     dim_out = 128
#     batch_size = 2
#
#     time_frames = 100
#     nb_bands_per_split = (24, 12, 8, 8, 8, 2)
#     freq_bins_per_band_per_split = (2, 4, 12, 24, 48, 128)
#     nb_bands = sum(nb_bands_per_split)
#     freq_bins = sum(nb_bands*freq_bins for (nb_bands, freq_bins) in zip(nb_bands_per_split, freq_bins_per_band_per_split) )
#     print(f'nb_bands: {nb_bands}, freq_bins: {freq_bins}')
#
#     bandsplit = BandSplit(dim_in, dim_out, nb_bands_per_split, freq_bins_per_band_per_split)
#     x = torch.randn(batch_size, dim_in, freq_bins, time_frames)
#     bandsplit.to('cuda')
#     x = x.to('cuda')
#
#     out = bandsplit(x)
#     print(f'out.shape: {out.shape}')
#     assert out.shape == (batch_size, dim_out, nb_bands, time_frames)