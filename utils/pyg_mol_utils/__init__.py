from .removehs import *
from torch_geometric.utils import to_undirected


def make_graph_undirected(data):
    edge_index = data.edge_index

    undirected_edge_index = to_undirected(edge_index)

    return Data(data.x, edge_index=undirected_edge_index, pos=data.pos)  # type: ignore


def add_master_node(data):
    num_nodes = data.num_nodes
    edge_index = data.edge_index

    src_edge = torch.arange(num_nodes, dtype=edge_index.dtype, device=edge_index.device)
    dest_edge = torch.tensor(num_nodes, dtype=edge_index.dtype, device=edge_index.device).expand(
        (num_nodes,)
    )

    edge_index_for_new_node = torch.stack([src_edge, dest_edge], dim=0)

    new_edge_index = torch.cat((edge_index, edge_index_for_new_node), dim=-1)

    new_x = torch.cat((data.x, torch.tensor([0], dtype=data.x.dtype, device=data.x.device)), dim=0)

    return Data(new_x, new_edge_index)  # type: ignore


__all__ = ["removeHs", "add_master_node"]
