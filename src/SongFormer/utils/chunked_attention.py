import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


DEFAULT_CHUNK_SIZE = 1024


def chunked_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> torch.Tensor:
    """Compute exact SDPA in query-axis chunks to reduce peak memory."""
    _, _, seq_len, head_dim = query.shape
    scale = 1.0 / math.sqrt(head_dim)
    output = torch.empty_like(query)

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        q_chunk = query[:, :, start:end, :]
        scores = torch.matmul(q_chunk, key.transpose(-2, -1)) * scale

        if attn_mask is not None:
            if attn_mask.dim() == 4 and attn_mask.shape[2] > 1:
                scores = scores + attn_mask[:, :, start:end, :]
            else:
                scores = scores + attn_mask

        attn_weights = F.softmax(scores, dim=-1)
        if dropout_p > 0.0 and query.requires_grad:
            attn_weights = F.dropout(attn_weights, p=dropout_p)
        output[:, :, start:end, :] = torch.matmul(attn_weights, value)

    return output


def _make_chunked_forward(original_forward, chunk_size: int):
    self_attention = original_forward.__self__

    def chunked_forward(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        relative_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, _, _ = hidden_states.size()
        query_key_states = hidden_states
        value_states = hidden_states

        if self_attention.position_embeddings_type == "rotary":
            if relative_position_embeddings is None:
                raise ValueError("relative_position_embeddings required for rotary mode")
            query_key_states = self_attention._apply_rotary_embedding(
                query_key_states, relative_position_embeddings
            )

        query = self_attention.linear_q(query_key_states).view(
            batch_size, -1, self_attention.num_heads, self_attention.head_size
        )
        key = self_attention.linear_k(query_key_states).view(
            batch_size, -1, self_attention.num_heads, self_attention.head_size
        )
        value = self_attention.linear_v(value_states).view(
            batch_size, -1, self_attention.num_heads, self_attention.head_size
        )

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        dropout_p = self_attention.dropout_p if self_attention.training else 0.0
        attn_output = chunked_scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            chunk_size=chunk_size,
        )

        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, -1, self_attention.num_heads * self_attention.head_size
        )
        attn_output = self_attention.linear_out(attn_output)
        return attn_output, None

    return chunked_forward


def apply_chunked_attention(conformer, chunk_size: int = DEFAULT_CHUNK_SIZE):
    """Patch conformer attention layers to chunked exact attention."""
    patched_count = 0
    for layer in conformer.layers:
        attn_module = getattr(layer, "self_attn", None)
        if attn_module is None:
            for attr_name in ("attention", "self_attention"):
                attn_module = getattr(layer, attr_name, None)
                if attn_module is not None:
                    break
        if attn_module is None:
            continue

        attn_module.forward = _make_chunked_forward(attn_module.forward, chunk_size)
        patched_count += 1
    return patched_count
