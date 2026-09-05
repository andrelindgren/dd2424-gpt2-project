import math

import torch

from einops import rearrange
from torch import nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):
    # key, query, value: [bs, h, t, d]
    # h = num_attention_heads
    # t = seq_len
    # d = attention_head_size
    seq_len = key.size(2)

    # scores = Q @ K^T / sqrt(d)
    # Transpose: [bs, h, t, d] -> [bs, h, d, t]
    # [bs, h, t, d] @ [bs, h, d, t] = [bs, h, t, t]
    scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.attention_head_size)

    # Causal mask: position i may only attend to positions <= i
    # Upper triangle: [t, t]
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=scores.device), diagonal=1)

    # Masked slots with -inf will give softmax roughly 0
    # Have to use dtype min and not -inf because of problems with NaN
    # [bs, h, t, t] + [t, t] = [bs, h, t, t]
    scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)

    # Attention padding mask adds large negative number which give softmax roughly 0
    # [bs, h, t, t] + [bs, 1, 1, t] = [bs, h, t, t]
    scores = scores + attention_mask

    # softmax over keys (last dim)
    # [bs, h, t, t] -> [bs, h, t, t]
    attn_probs = F.softmax(scores, dim=-1)

    # Dropout for attention, doesn't change shape
    # [bs, h, t, t] -> [bs, h, t, t]
    attn_probs = self.dropout(attn_probs)

    # attn_value = P @ V:
    # [bs, h, t, t] @ [bs, h, t, d] = [bs, h, t, d]
    attn_value = torch.matmul(attn_probs, value)

    # Rearrange
    # [bs, h, t, d] -> [bs, t, h, d]
    attn_value = rearrange(attn_value, 'b h t d -> b t h d')
    # [bs, t, h, d] -> [bs, t, h*d]
    attn_value = rearrange(attn_value, 'b t h d -> b t (h d)')

    # [bs, t, hidden_size]
    return attn_value


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
