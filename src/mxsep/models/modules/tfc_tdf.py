from torch import nn
from einops import einops

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight_freeze=False, bias_freeze=False, weight_init=1.0,
                 bias_init=0.0, **kwargs):
        super().__init__(num_features, eps=eps, momentum=momentum)
        if weight_init is not None: nn.init.constant_(self.weight, weight_init)
        if bias_init is not None: nn.init.constant_(self.bias, bias_init)
        self.weight.requires_grad_(not weight_freeze)
        self.bias.requires_grad_(not bias_freeze)


def get_norm(norm, out_channels, **kwargs) -> nn.Module:
    """
    Args:
        norm (str or callable): either one of BN, GhostBN, FrozenBN, GN or SyncBN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module
        out_channels: number of channels for normalization layer

    Returns:
        nn.Module or None: the normalization layer
    """
    # return nn.BatchNorm2d(out_channels)

    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": BatchNorm,
        }[norm]
    return norm(out_channels, **kwargs)


class TFC(nn.Module):
    def __init__(self, c_in, c_out, l, k, bn_norm):
        super(TFC, self).__init__()

        self.H = nn.ModuleList()
        for i in range(l):
            if i == 0:
                c_in = c_in
            else:
                c_in = c_out
            self.H.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=k, stride=1, padding=k // 2),
                    get_norm(bn_norm, c_out),
                    nn.ReLU(),
                )
            )

    def forward(self, x):
        """
        :param x: Input tensor with shape: (batch, c_in, f, t)
        :return: Output Tensor with shape (batch, c_out, f, t)
        """
        for h in self.H:
            x = h(x)
        return x
        return x


class TFC_TDF_Res2(nn.Module):
    def __init__(self, c_in, c_out, l, f, k, bn, bn_norm, bias=True):

        super(TFC_TDF_Res2, self).__init__()

        self.use_tdf = bn is not None

        self.tfc1 = TFC(c_in, c_out, l, k, bn_norm)
        self.tfc2 = TFC(c_in, c_out, l, k, bn_norm)

        self.res = TFC(c_in, c_out, 1, k, bn_norm)

        if self.use_tdf:
            if bn == 0:
                # print(f"TDF={f},{f}")
                self.tdf = nn.Sequential(
                    nn.Linear(f, f, bias=bias),
                    get_norm(bn_norm, c_out),
                    nn.ReLU()
                )
            else:
                # print(f"TDF={f},{f // bn},{f}")
                self.tdf = nn.Sequential(
                    nn.Linear(f, f // bn, bias=bias),
                    get_norm(bn_norm, c_out),
                    nn.ReLU(),
                    nn.Linear(f // bn, f, bias=bias),
                    get_norm(bn_norm, c_out),
                    nn.ReLU()
                )

    def forward(self, x):
        """
        :param x: Input tensor with shape: (batch, c_in, f, t)
        :return: Output Tensor with shape (batch, c_out, f, t)
        """

        res = self.res(x)
        x = self.tfc1(x)
        if self.use_tdf:
            x = einops.rearrange(x, 'b c f t -> b c t f')
            x = x + self.tdf(x)
            x = einops.rearrange(x, 'b c t f -> b c f t')
        x = self.tfc2(x)
        x = x + res
        return x


class DPTDFFirstConv(nn.Module):

    def __init__(self, c_in, c_out, bn_norm):
        super().__init__()

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1)),
            get_norm(bn_norm, c_out),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        :param x: Input tensor with shape: (batch, c_in, f, t)
        :return: Output Tensor with shape (batch, c_in, f, t)
        """
        x = self.first_conv(x)
        return x


class DPTDFFinalConv(nn.Module):

    def __init__(self, c_in, c_out):
        super().__init__()

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1)),
        )

    def forward(self, x):
        """
        :param x: Input tensor with shape: (batch, c_in, f, t)
        :return: Output Tensor with shape (batch, c_in, f, t)
        """
        x = self.final_conv(x)
        return x


class DPTDFEncoderBlock(nn.Module):

    def __init__(self, c_in, c_out, l, f, k, bn, bn_norm, bias=True):
        super().__init__()
        scale = (2, 2)

        self.tfc_tdf = TFC_TDF_Res2(c_in, c_in, l, f, k, bn, bn_norm, bias)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=scale, stride=scale),
            get_norm(bn_norm, c_out),
            nn.ReLU()
        )

    def forward(self, x):
        """
        :param x: Input tensor with shape: (batch, c_in, f, t)
        :return: Output Tensor with shape (batch, c_in, f, t),
                 Residual tensor with shape: (batch, c_out, f//2, t//2)
        """
        x = self.tfc_tdf(x)
        res = x
        x = self.downsample(x)
        return x, res


class DPTDFDecoderBlock(nn.Module):

    def __init__(self, c_in, c_out, l, f, k, bn, bn_norm, bias=True):
        super().__init__()
        scale = (2, 2)

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=c_in, out_channels=c_out, kernel_size=scale, stride=scale),
            get_norm(bn_norm, c_out),
            nn.ReLU()
        )
        self.tfc_tdf = TFC_TDF_Res2(c_out, c_out, l, f, k, bn, bn_norm, bias)

    def forward(self, x, res):
        """
        :param x: Input tensor with shape  = (batch, c_in, f//2, t//2)
        :param res: Residual tensor with shape = (batch, c_out, f, t)
        :return: Tensor with shape (batch, c_out, f, t)
        """
        x = self.upsample(x)
        x = x * res
        x = self.tfc_tdf(x)
        return x
