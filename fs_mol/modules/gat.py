import torch
from torch import nn
from torch_geometric.nn.models import GAT
from torch_geometric.nn.aggr import SumAggregation
from dataclasses import dataclass

@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 32
    graph_encoder_num_layers: int = 5
    graph_encoder_hidden_dim: int = 80
    graph_encoder_out_dim: int = 256
    graph_encoder_heads: int = 4
    graph_encoder_edge_dim: int = 1
    graph_encoder_dropout: float = 0.1
    graph_encoder_mlp_hidden_dim: int = 512
    
    fingerprint_encoder_hidden_dim: int = 1024
    fingerprint_encoder_output_dim: int = 512
    fingerprint_encoder_dropout: int = 0.1
    

class GAT_GraphEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gnn = GAT(
            32,
            hidden_channels=config.graph_encoder_hidden_dim,
            num_layers=config.graph_encoder_num_layers,
            out_channels=config.graph_encoder_out_dim,
            heads=config.graph_encoder_heads,
            v2=True,
            edge_dim=config.graph_encoder_edge_dim,
            dropout=config.graph_encoder_dropout,
            add_self_loops=True,
        )
        
        self.aggr = SumAggregation()
        
        mlp_hidden_dim = config.fingerprint_encoder_hidden_dim
        mlp_output_dim = config.fingerprint_encoder_output_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(config.graph_encoder_out_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(mlp_hidden_dim),
            nn.Linear(mlp_hidden_dim, mlp_output_dim)
        )
        
    def forward(self, batch):
        node_feats = self.gnn(batch.x, batch.edge_index, edge_attr=batch.edge_attr.to(torch.float32))
        graph_feats = self.aggr(node_feats, batch.batch)
        
        return self.mlp(graph_feats)