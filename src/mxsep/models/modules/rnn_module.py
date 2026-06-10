import torch
import torch.nn as nn


class RNNModule(nn.Module):
    """
    RNN submodule of BandSequence module
    """

    def __init__(
            self,
            group_num: int,
            input_dim_size: int,
            hidden_dim_size: int,
            rnn_type: str = 'LSTM',
            bidirectional: bool = True
    ):
        super(RNNModule, self).__init__()
        self.group_norm = nn.GroupNorm(group_num, input_dim_size)
        self.rnn = getattr(nn, rnn_type)(
            input_dim_size, hidden_dim_size, batch_first=True, bidirectional=bidirectional # The output is  2 * hidden_dim_size，because it is bi
        )
        self.fc = nn.Linear(
            hidden_dim_size * 2 if bidirectional else hidden_dim_size,
            input_dim_size
        )

    def forward(
            self,
            x: torch.Tensor
    ):
        """
        Input shape:
            across T - [batch_size x k_subbands, time, n_features]
            OR
            across K - [batch_size x time, k_subbands, n_features]
        """
        out = self.group_norm(
            x.transpose(-1, -2)
        ).transpose(-1, -2)
        out = self.rnn(out)[0]  # The last dimension is the feature.
        out = self.fc(out)

        return out + x