from typing import Tuple

import torch
from einops import einops
from torch import nn
from mxsep.models.modules.transformer import RMSNorm


class MaskEstimator(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        nb_bands_per_split: Tuple[int, ...] = (24, 12, 8, 8, 8, 2),
        freq_bins_per_band_per_split: Tuple[int, ...] = (2, 4, 12, 24, 48, 128),
        mlp_expansion_factor: int = 4
    ):
        super().__init__()
        assert len(nb_bands_per_split) == len(freq_bins_per_band_per_split)
        self.nb_bands_per_split = tuple(nb_bands_per_split)
        self.freq_bins_per_band_per_split = freq_bins_per_band_per_split
        self.mlp_expansion_factor = mlp_expansion_factor
        self.rmsnorm = RMSNorm(dim_in) # todo Should we use RMSNorm with learned weights per band as in Original impl?
        
        self.total_nb_bands = sum(nb_bands_per_split)

        self.fc1_weights = nn.Parameter(torch.randn(self.total_nb_bands, dim_in * mlp_expansion_factor, dim_in))
        nn.init.kaiming_uniform_(self.fc1_weights)
        self.fc1_bias = nn.Parameter(torch.randn(self.total_nb_bands, dim_in * mlp_expansion_factor))
        nn.init.normal_(self.fc1_bias, std=0.02)
        self.tanh = nn.Tanh()
        self.glu = nn.GLU()
        
        self.fc2_weights_per_split = nn.ParameterList([])
        self.fc2_bias_per_split = nn.ParameterList([]) 

        for nb_bands, freq_bins in zip(
            nb_bands_per_split, freq_bins_per_band_per_split
        ):
            weights = nn.Parameter(torch.randn(nb_bands, 2*freq_bins*dim_out, dim_in * mlp_expansion_factor))
            nn.init.kaiming_uniform_(weights)
            bias = nn.Parameter(torch.randn(nb_bands,  2*freq_bins*dim_out))
            nn.init.normal_(bias, std=0.02)
            self.fc2_weights_per_split.append(weights)
            self.fc2_bias_per_split.append(bias)

    def forward(self, x):
        """

        :type x: Tensor with shape  (batch_size, dim_in, nb_bands, time_frames)

        :return masks: tensor with shape (batch_size, dim_out, freq_bins, time_frames)
        """

        x = einops.rearrange(x, "b d n t -> b t n d")
        x = self.rmsnorm(x)
        x = torch.einsum("btni,noi->btno", x, self.fc1_weights)
        x = x + self.fc1_bias.unsqueeze(0).unsqueeze(0)
        x = self.tanh(x)

        x = x.split(self.nb_bands_per_split, dim=-2)

        outs = []
        for x_i, w_i, b_i, n_i, f_i in zip(
            x,
            self.fc2_weights_per_split,
            self.fc2_bias_per_split,
            self.nb_bands_per_split,
            self.freq_bins_per_band_per_split,
        ):
            x_i = torch.einsum("btni,noi->btno", x_i, w_i)
            x_i = x_i + b_i.unsqueeze(0).unsqueeze(0)
            x_i = self.glu(x_i)
            
            x_i = einops.rearrange(x_i, "b t n (f d) -> b d (n f) t ", n=n_i, f=f_i)
            outs.append(x_i)

        return torch.cat(outs, dim=-2)
        
        
# if __name__ == '__main__':
#     dim_in = 128
#     dim_out = 4
#     batch_size = 2
#
#     time_frames = 100
#     nb_bands_per_split = (24, 12, 8, 8, 8, 2)
#     freq_bins_per_band_per_split = (2, 4, 12, 24, 48, 128)
#     nb_bands = sum(nb_bands_per_split)
#     freq_bins = sum(
#         nb_bands * freq_bins
#         for (nb_bands, freq_bins) in zip(
#             nb_bands_per_split, freq_bins_per_band_per_split
#         )
#     )
#     print(f"nb_bands: {nb_bands}, freq_bins: {freq_bins}")
#
#     mask_estimator = MaskEstimator(
#         dim_in, dim_out, nb_bands_per_split, freq_bins_per_band_per_split
#     )
#     x = torch.randn(batch_size, dim_in, nb_bands, time_frames)
#     mask_estimator.to("cuda")
#     x = x.to("cuda")
#
#     out = mask_estimator(x)
#     print(f"out.shape: {out.shape}")
#     assert out.shape == (batch_size, dim_out, freq_bins, time_frames)
