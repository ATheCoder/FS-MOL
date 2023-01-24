from typing import Optional, Tuple

from torch import nn
import torch

from fs_mol.models.graphormer.graph_node_feature import GraphAttnBias, GraphNodeFeature

from .graphormer_layer import GraphormerEncoderLayer

class GraphormerGraphEncoder(nn.Module):
    def __init__(
        self,
        num_atoms: int,
        num_in_degree: int,
        num_out_degree: int,
        num_edges: int,
        num_spatial: int,
        num_edge_dis: int,
        edge_type: int,
        multi_hop_max_dist: int,
        num_encoder_layers: int = 12,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 768,
        num_attention_heads: int = 32,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        layerdrop: float = 0.0,
        encoder_normalize_before: bool = False,
        pre_layernorm: bool = False,
        apply_graphormer_init: bool = False,
        activation_fn: str = "gelu",
        embed_scale: float = None,
        freeze_embeddings: bool = False,
        n_trans_layers_to_freeze: int = 0,
        export: bool = False,
        traceable: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8
    ) -> None:
        super().__init__()
        
        self.dropout_module = nn.Dropout(dropout)
        
        self.layerdrop = layerdrop
        self.embedding_dim = embedding_dim
        self.apply_graphormer_init = apply_graphormer_init
        self.traceable = traceable
        
        self.graph_node_features = GraphNodeFeature(
            num_heads=num_attention_heads,
            num_atoms=num_atoms,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers
        )
        
        self.graph_attn_bias = GraphAttnBias(
            num_heads=num_attention_heads,
            num_atoms=num_atoms,
            num_edges=num_edges,
            num_spatial=num_spatial,
            num_edge_dis=num_edge_dis,
            edge_type=edge_type,
            multi_hop_max_dist=multi_hop_max_dist,
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers
        )
        
        self.embed_scale = embed_scale
        
        if q_noise > 0:
            pass
        else:
            pass
        
        if encoder_normalize_before:
            self.embed_layer_norm = nn.LayerNorm(self.embedding_dim)
        else:
            self.embed_layer_norm = None
        
        if pre_layernorm:
            self.final_layer_norm = nn.LayerNorm(self.embedding_dim)
            
        if self.layerdrop > 0.0:
            pass
            # TODO: Layer Drop
            # self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])
        
        self.layers.extend(
            [
                GraphormerEncoderLayer(
                    embedding_dim=embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    export=export,
                    q_noise=q_noise,
                    qn_block_size=qn_block_size,
                    pre_layernorm=pre_layernorm,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        
        # NOTE: Graphormer init is not implemented
        # if self.apply_graphormer_init:
        #     self.apply(init)
        
        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False
        
        for layer in range(n_trans_layers_to_freeze):
            freeze_module_params(self.layers[layer])
            
    def forward(
        self,
        batched_data,
        perturb=None,
        last_state_only: bool = False,
        token_embedding: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        data_x = batched_data["x"]
        
        n_graph, n_node = data_x.size()[:2]
        
        padding_mask = (data_x[:, :, 0]).eq(0)
        padding_mask_cls = torch.zeros(
            n_graph, 1, device=padding_mask.device, dtype=padding_mask.dtype
        )
        
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)
        
        # TODO: Graph Node Feature
        if token_embedding is not None:
            x = token_embedding
        else:
            x = self.graph_node_feature(batched_data)
        
        # TODO: Attention Bias
        attn_bias = self.graph_attn_bias(batched_data)
        
        # NOTE: Embed Scale
        # NOTE: Quant Noise
        # NOTE: Embed Layer Norm
        
        x = self.dropout_module(x)
        
        x = x.transpose(0, 1)
        
        inner_states = []
        if not last_state_only:
            inner_states.append(x)
        
        for layer in self.layers:
            x, _ = layer(
                x,
                self_attn_padding_mask=padding_mask,
                self_attn_mask=attn_mask,
                self_attn_bias=attn_bias
            )
            
            if not last_state_only:
                inner_states.append(x)
                
        graph_repr = x[0, :, :]
        
        if last_state_only:
            inner_states = [x]
        
        if self.traceable:
            return torch.stack(inner_states), graph_repr
        else:
            return inner_states, graph_repr