from typing import Callable

from torch import nn
from torch.nn import functional as F


def get_activation_fn(activation_name: str):
    if activation_name == 'relu':
        return F.relu


class GraphormerEncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        export: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        init_fn: Callable = None,
        pre_layernorm: bool = False
    ) -> None:
        super().__init__()
        
        if init_fn is not None:
            init_fn()
            
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.q_noise = q_noise
        self.qn_block_size = qn_block_size
        self.pre_layernorm = pre_layernorm
        
        self.dropout_module = nn.Dropout(dropout)
        
        self.activation_dropout_module = nn.Dropout(activation_dropout)
        
        self.activation_fn = get_activation_fn(activation_fn)
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            # TODO: PyTorch doesn't accept q_noise do we need this?
            # TODO: qn_block_size also has to do with `quant_noise`
        )
        
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)
        
        # TODO: Quant Noise is applicable on both FC Layers.
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)
        
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)
        
    def forward(self, x, self_attn_bias, self_attn_mask, self_attn_padding_mask):
        
        residual = x
        
        if self.pre_layernorm:
            x = self.self_attn_layer_norm(x)
        
        x = self.self_attn(
            x,
            x,
            x,
            # TODO: attn_bias is not possible using PyTorch's Implementation of MultiHeadAttention
            key_padding_mask=self_attn_padding_mask,
            attn_mask=self_attn_mask
        )
        
        x = self.dropout_module(x)
        
        x = residual + x
        
        if not self.pre_layernorm:
            x = self.self_attn_layer_norm(x)
            
        residual = x
        
        if self.pre_layernorm:
            x = self.final_layer_norm(x)
            
        x = self.activation_fn(self.fc1(x))
        
        x = self.activation_dropout_module(x)
        
        x = self.fc2(x)
        
        x = self.dropout_module(x)
        x = residual + x
        if not self.pre_layernorm:
            x = self.final_layer_norm(x)
            
        return x