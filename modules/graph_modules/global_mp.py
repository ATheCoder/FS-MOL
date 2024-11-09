import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing, SetTransformerAggregation, GINConv
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.utils import add_self_loops


from modules.mxm import MLP, Res, Xavier_SiLU_MLP, Xavier_SiLU_Residual


class ResidualGIN(nn.Module):
    def __init__(self, dim, dropout=0.0) -> None:
        super().__init__()
        self.dim = dim
        self.graph_norm = GraphNorm(self.dim)
        self.linear = nn.Linear(self.dim, self.dim)
        self.gin_conv = GINConv(
            nn.Sequential(
                nn.Linear(self.dim, self.dim),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(self.dim, self.dim),
            )
        )

    def forward(self, h, edge_index, batch):
        h = self.graph_norm(h, batch)

        res_h = h

        h = self.gin_conv(h, edge_index)

        return h + res_h


class Global_MP_Attn(MessagePassing):
    def __init__(self, dim, out_dim, edge_dim, dropout=0.0, layer_no=0):
        super(Global_MP_Attn, self).__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.layer_no = layer_no

        self.h_mlp = MLP([self.dim, self.dim])

        self.res1 = Res(self.dim, dropout)
        self.res2 = Res(self.dim, dropout)
        self.res3 = Res(self.dim, dropout)
        self.mlp = MLP([self.dim, self.dim], dropout)

        self.x_edge_mlp = MLP([2 * self.dim + self.edge_dim, self.dim], dropout)
        self.linear = MLP([self.dim, self.dim])
        self.norm = GraphNorm(dim)
        self.last_layer_projector = nn.Linear(self.dim, self.dim)
        self.aggr_out_mlp = MLP([self.dim, self.dim], dropout)
        self.layer_norm = GraphNorm(self.dim)
        self.layer_norm_2 = GraphNorm(self.dim)

        self.aggregator = SetTransformerAggregation(self.dim, heads=4)

    def forward(self, h, edge_attr, edge_index, batch):
        # edge_index, edge_attr = sort_edge_index(edge_index, edge_attr)
        # edge_index, _ = add_self_loops(edge_index, num_nodes=h.size(0))
        h = self.layer_norm_2(h, batch)
        res_h = h
        # h = self.norm(h, batch)

        # h = self.norm(h, batch)
        # Message Passing operation
        h = self.propagate(edge_index, x=h, num_nodes=h.size(0), edge_attr=edge_attr, batch=batch)

        # # # Update function f_u
        # h = self.res1(h, batch)

        # h = h + res_h

        # h = self.norm(h)

        # h = self.res2(h, batch)

        # h = self.res3(h, batch)

        # # # Message Passing operation
        # h = self.propagate(edge_index, x=h, num_nodes=h.size(0), edge_attr=edge_attr, batch=batch)
        return h + res_h
        # h = h + res_h

        # h = self.layer_norm(h, batch)
        # h_b = self.bottleneck(h)

        # return h + h_b

    def message(self, x_i, x_j, edge_attr, edge_index, num_nodes, batch):
        # num_edge = edge_attr.size()[0]

        # x_edge = torch.cat((x_i[:num_edge], x_j[:num_edge], edge_attr), -1)
        # x_edge = self.x_edge_mlp(x_edge)
        # wandb.log(
        #     {f"before_cat_{self.layer_no}": wandb.Histogram(x_edge.reshape(-1).cpu().detach())}
        # )

        # x_j = self.linear(edge_attr) * x_edge

        # wandb.log({f"after_cat_{self.layer_no}": wandb.Histogram(x_j.reshape(-1).cpu().detach())})

        return x_j

    def aggregate(
        self, inputs: Tensor, index: Tensor, ptr: Tensor | None = None, dim_size: int | None = None
    ) -> Tensor:
        h = self.aggregator(inputs, index, ptr, dim_size)
        return h

    def update(self, aggr_out, batch):
        # aggr_out = self.norm(aggr_out, batch)
        return aggr_out


class Global_MP(MessagePassing):
    def __init__(self, dim, out_dim, edge_dim, dropout=0.0, layer_no=0):
        super(Global_MP, self).__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.layer_no = layer_no

        self.h_mlp = MLP([self.dim, self.dim])

        self.res1 = Res(self.dim, dropout)
        self.res2 = Res(self.dim, dropout)
        self.res3 = Res(self.dim, dropout)
        self.mlp = MLP([self.dim, self.dim], dropout)

        self.x_edge_mlp = MLP([2 * self.dim + self.edge_dim, self.dim], dropout)
        self.linear = MLP([self.dim, self.dim])
        self.norm = GraphNorm(dim)
        self.last_layer_projector = nn.Linear(self.dim, self.dim)
        self.aggr_out_mlp = MLP([self.dim, self.dim], dropout)
        self.layer_norm = GraphNorm(self.dim)
        self.layer_norm_2 = GraphNorm(self.dim)

        self.bottleneck = nn.Sequential(
            nn.Linear(self.dim, 4 * self.dim), nn.LeakyReLU(), nn.Linear(4 * self.dim, self.dim)
        )

        self.aggregator = SetTransformerAggregation(self.dim, heads=4, layer_norm=True)

    def forward(self, h, edge_attr, edge_index, batch):
        # edge_index, edge_attr = sort_edge_index(edge_index, edge_attr)
        # edge_index, _ = add_self_loops(edge_index, num_nodes=h.size(0))
        h = self.layer_norm_2(h, batch)
        res_h = h
        # h = self.norm(h, batch)

        # h = self.norm(h, batch)
        # Message Passing operation
        h = self.propagate(edge_index, x=h, num_nodes=h.size(0), edge_attr=edge_attr, batch=batch)

        # # # Update function f_u
        # h = self.res1(h, batch)

        # h = h + res_h

        # h = self.norm(h)

        # h = self.res2(h, batch)

        # h = self.res3(h, batch)

        # # # Message Passing operation
        # h = self.propagate(edge_index, x=h, num_nodes=h.size(0), edge_attr=edge_attr, batch=batch)
        # return h + res_h
        h = h + res_h

        h = self.layer_norm(h, batch)
        h_b = self.bottleneck(h)

        return h + h_b

    def message(self, x_i, x_j, edge_attr, edge_index, num_nodes, batch):
        num_edge = edge_attr.size()[0]

        x_edge = torch.cat((x_i[:num_edge], x_j[:num_edge], edge_attr), -1)
        x_edge = self.x_edge_mlp(x_edge)

        x_j = self.linear(edge_attr) * x_edge

        return x_j

    # def aggregate(self, inputs: Tensor, index: Tensor, ptr: Tensor | None = None, dim_size: int | None = None) -> Tensor:
    #     h = self.aggregator(inputs, index, ptr, dim_size)
    #     return h

    def update(self, aggr_out, batch):
        # aggr_out = self.norm(aggr_out, batch)
        return aggr_out


class Generic_Global_MP(MessagePassing):
    def __init__(self, dim, out_dim, edge_dim, dropout=0.0, attn_dropout=0.0):
        super(Generic_Global_MP, self).__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim

        self.h_mlp = Xavier_SiLU_MLP([self.dim, self.dim])
        self.mlp = Xavier_SiLU_MLP([self.dim, self.dim], dropout)
        self.x_edge_mlp = Xavier_SiLU_MLP([2 * self.dim + self.edge_dim, self.dim], dropout)

        self.res1 = Xavier_SiLU_Residual(self.dim, dropout)
        self.res2 = Xavier_SiLU_Residual(self.dim, dropout)
        self.res3 = Xavier_SiLU_Residual(self.dim, dropout)

        self.linear = nn.Linear(self.dim, self.dim, bias=False)
        self.norm = GraphNorm(dim)

        self.last_layer_projector = nn.Linear(self.dim, self.dim)

    def forward(self, h, edge_attr, edge_index, batch):
        edge_index, _ = add_self_loops(edge_index, num_nodes=h.size(0))

        res_h = h

        # Message Passing operation
        h = self.propagate(edge_index, x=h, num_nodes=h.size(0), edge_attr=edge_attr, batch=batch)

        # Update function f_u
        h = self.res1(h, batch)

        h = self.mlp(h) + res_h

        h = self.res2(h, batch)

        h = self.res3(h, batch)

        # Message Passing operation
        h = self.propagate(edge_index, x=h, num_nodes=h.size(0), edge_attr=edge_attr, batch=batch)

        h = self.h_mlp(self.norm(h, batch))

        return h

    def message(self, x_i, x_j, edge_attr, edge_index, num_nodes, batch):
        num_edge = edge_attr.size()[0]

        x_edge = torch.cat((x_i[:num_edge], x_j[:num_edge], edge_attr), -1)
        x_edge = self.x_edge_mlp(x_edge)

        x_j = torch.cat((self.linear(edge_attr) * x_edge, x_j[num_edge:]), dim=0)

        return x_j

    def update(self, aggr_out, batch):
        return aggr_out
