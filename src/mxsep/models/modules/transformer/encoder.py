# see https://github.com/pytorch/xla/blob/master/examples/decoder_only_model.py
# see http://github.com/meta-llama/llama-models/blob/main/models/llama3/model.py
# see https://waylandz.com/llm-transformer-book-en/chapter-25-positional-encoding-evolution/
# see https://towardsdatascience.com/positional-embeddings-in-transformers-a-math-guide-to-rope-alibi/

import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


def apply_xla_flash_attention(query_states, key_states, value_states):
  from torch_xla.experimental.custom_kernel import flash_attention

  # q, k, v should all have the shape [B, n_head, S, head_dim]
  attn_output = flash_attention(
      query_states, key_states, value_states, causal=False)
  return attn_output


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def apply_scaling(freqs: torch.Tensor) -> torch.Tensor:
    # Values obtained from grid search
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * torch.pi / freqs
    new_freqs = torch.where(wavelen > low_freq_wavelen, freqs / scale_factor, freqs)
    smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    return torch.where(
        (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen),
        (1 - smooth) * new_freqs / scale_factor + smooth * new_freqs,
        new_freqs,
    )

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def get_alibi_slopes(n_heads: int) -> torch.Tensor:
    start = 2 ** (-8 / n_heads)
    ratio = start
    return torch.tensor([start * (ratio ** i) for i in range(n_heads)])


def get_linear_bias(n_heads: int, ctx_size:int) -> torch.Tensor:
    slopes = get_alibi_slopes(n_heads).view(n_heads, 1, 1)
    pos = torch.arange(ctx_size)
    distances = pos[None, :] - pos[:, None]
    #  For encoder we use -abs instead of min(0, distance)
    distances = - distances.abs()
    distances.unsqueeze_(0)  # 1, ctx_size, ctx_size
    linear_bias = distances * slopes  # n_heads, ctx_size, ctx_size
    return linear_bias.unsqueeze(0)  # 1, n_heads, ctx_size, ctx_size

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to LlamaRMSNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# 1. no kv_chche
# 2. no rotary embedding
# 3. no attention_mask
class GroupQueryAttention(nn.Module):
    """Stripped-down version of the LlamaAttention"""

    def __init__(
        self,
        hidden_size: int = 1024,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        use_flash_attention=False,
        pos_embedding:str='rope',
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.use_flash_attention = use_flash_attention

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        if self.use_flash_attention:
            #todo check if using xla
            self.flash_attention_impl = apply_xla_flash_attention
        else:
            self.flash_attention_impl = None

        self.pos_embedding = pos_embedding
        if self.pos_embedding == 'alibi':
            linear_bias = get_linear_bias(n_heads=self.num_heads, ctx_size=512)  # todo share overlayer (~freqs_cis)
            self.register_buffer("linear_bias", linear_bias)
            

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()
        # [B, S, H] -> [B, S, n_head * head_dim]
        query_states = self.q_proj(hidden_states)
        # [B, S, H] -> [B, S, n_kv_head * head_dim]
        key_states = self.k_proj(hidden_states)
        # [B, S, H] -> [B, S, n_kv_head * head_dim]
        value_states = self.v_proj(hidden_states)

        # [B, S, n_head * head_dim] -> [B, S, n_head,  head_dim]
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        )
        # [B, S, n_kv_head * head_dim] -> [B, S, n_kv_head, head_dim]
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )
        if self.pos_embedding == 'rope':
            query_states, key_states = apply_rotary_emb(query_states, key_states, freqs_cis=freqs_cis)

        # [B, S, n_head,  head_dim] -> [B, n_head, S, head_dim]
        query_states = query_states.transpose(1, 2)
        # [B, S, n_kv_head, head_dim] -> [B, n_kv_head, S, head_dim]
        key_states = key_states.transpose(1, 2)
        
        # [B, S, n_kv_head * head_dim] -> [B, n_kv_head, S, head_dim]
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # [B, n_kv_head, S, head_dim] -> [B, n_head, S, head_dim]
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        # [B, n_kv_head, S, head_dim] -> [B, n_head, S, head_dim]
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if not self.use_flash_attention:
            # [B, n_head, S, head_dim] @ T([B, n_head, S, head_dim]) -> [B, n_head, S, S]
            attn_weights = torch.einsum(
                "bnsh,bnkh->bnsk", query_states, key_states
            ) / math.sqrt(self.head_dim)

            if self.pos_embedding == "alibi":
                linear_bias = self.linear_bias[:, :, :q_len, :q_len]
                attn_weights = attn_weights + linear_bias

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)

            # [B, n_head, S, S] @ T([B, n_head, S, head_dim]) -> [B, n_head, S, head_dim]
            attn_output = torch.einsum("bnsk,bnkh->bnsh", attn_weights, value_states)
        else:
            assert self.flash_attention_impl != None
            assert self.pos_embedding != "alibi"
            # [B, n_head, S, head_dim], [B, n_head, S, head_dim], [B, n_head, S, head_dim]
            # -> [B, n_head, S, head_dim]
            attn_output = self.flash_attention_impl(
                query_states, key_states, value_states
            )

        # [B, n_head, S, head_dim] -> [B * S * n_head * head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [B * S * n_head * head_dim] -> [B, S, H]
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # [B, S, H] -> [B, S, H]
        attn_output = self.o_proj(attn_output)

        return attn_output


class MLP(nn.Module):
    """
    Stripped-down version of the LlamaMLP
    SwiGLU implementation
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 3 * 1024,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.silu

    def forward(self, x):
        # [B, S, H] -> [B, S, I]
        up_proj = self.up_proj(x)
        # [B, S, H] -> [B, S, I]
        gate_proj = self.act_fn(self.gate_proj(x))
        # ([B, S, I] * [B, S, I]) -> [B, S, H]
        down_proj = self.down_proj(gate_proj * up_proj)
        return down_proj


class EncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int = 1024,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        intermediate_size: int = 3 * 1024,
        use_flash_attention=False,
        max_seq_len=2048,
        rope_theta: float = 10000.0,
        use_scaled_rope: bool = False,
        pos_embedding: str = 'rope',
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.self_attn = GroupQueryAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            use_flash_attention=use_flash_attention,
            pos_embedding=pos_embedding,
        )
        self.mlp = MLP(hidden_size=hidden_size, intermediate_size=intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)

        self.freqs_cis = precompute_freqs_cis(
            hidden_size // num_attention_heads,
            max_seq_len * 2,
            rope_theta,
            use_scaled_rope,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.FloatTensor:

        self.freqs_cis = self.freqs_cis.to(hidden_states.device)
        seqlen = hidden_states.shape[-2]
        freqs_cis = self.freqs_cis[0 : seqlen ]

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            freqs_cis = freqs_cis
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
