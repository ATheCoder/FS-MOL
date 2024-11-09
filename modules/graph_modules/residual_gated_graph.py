from typing import Any
import torch
from torch_geometric.nn import MessagePassing, SumAggregation
from torch import Tensor, nn


class ResGatedGNNLayer(MessagePassing):
    def __init__(self, dim: int, **kwargs):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

        self.u = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        self.a = nn.Linear(dim, dim)
        self.b = nn.Linear(dim, dim)

        self.sigmoid = nn.Sigmoid()

        self.relu = nn.LeakyReLU()

        self.batch_norm = nn.BatchNorm1d(dim)

    def forward(self, h, edge_index) -> Any:
        h = self.batch_norm(h)
        h_in = h

        h = self.propagate(edge_index, h=h)

        h = self.u(h_in) + h

        h = self.relu(h)

        return h + h_in

    def message(self, h_i: Tensor, h_j: Tensor):
        left = self.a(h_i)
        right = self.b(h_j)

        n_ij = self.sigmoid(left + right)

        return n_ij * self.v(h_j)


class ResidualGatedGraphEncoder(nn.Module):
    def __init__(
        self, init_node_embedding, eigen_vector_embedding, graph_embedding, n_layers
    ) -> None:
        super().__init__()

        dim = init_node_embedding

        self.message_passing = nn.ModuleList()

        self.node_embedding = nn.Embedding(16, init_node_embedding)

        # self.embed_eigen_vec = nn.Linear(eigen_vector_embedding, init_node_embedding)

        for _ in range(n_layers):
            self.message_passing.append(ResGatedGNNLayer(dim))

        # self.node_aggregator = SetTransformerAggregation(
        #     dim, heads=4, layer_norm=False, num_decoder_blocks=0
        # )
        
        self.node_aggregator = SumAggregation()

        self.final_mlp = nn.Linear(dim, graph_embedding)

    def forward(self, data):
        x, edge_index, eigen_vecs, batch = (
            data.x,
            data.edge_index,
            getattr(data, "laplacian_eigenvector_pe", None),
            data.batch,
        )

        x = self.node_embedding(x)
        # eigen_vecs = self.embed_eigen_vec(eigen_vecs)

        if eigen_vecs != None:
            h = torch.cat([x, eigen_vecs], dim=-1)
        else:
            h = x

        for layer in self.message_passing:
            h = layer(h, edge_index.long())

        graph_reprs = self.node_aggregator(h, batch)
        # graph_reprs = torch.nan_to_num(graph_reprs)
        out = self.final_mlp(graph_reprs)
        return out
