import torch
from torch_geometric.utils import k_hop_subgraph, to_undirected, subgraph
from torch_geometric.data import Data

def remove_k_hop_subgraph(data, k, node=None):
    # Ensure that the edge_index is undirected
    edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)

    # Select a random node if not provided
    if node is None:
        node = torch.randint(0, data.num_nodes, (1,)).item()

    # Get the nodes in the k-hop subgraph
    subgraph_nodes, _, _, _ = k_hop_subgraph(node, k, edge_index, relabel_nodes=False)

    # Invert the mask: True for nodes to keep, False for nodes to remove
    mask = torch.ones(data.num_nodes, dtype=torch.bool)
    mask[subgraph_nodes] = False

    # Get the new edge_index and mapping for nodes
    new_edge_index, new_mapping = subgraph(mask, edge_index, relabel_nodes=True)

    # Create a new Data object with the updated edge_index
    new_data = Data(x=data.x[mask] if data.x is not None else None, 
                    edge_index=new_edge_index,
                    edge_attr=data.edge_attr[new_mapping] if data.edge_attr is not None else None)

    # Copy other attributes
    for key, item in data:
        if key not in ['x', 'edge_index', 'edge_attr']:
            new_data[key] = item

    return new_data

# Example usage
# Assuming `data` is a PyTorch Geometric Data object
# data = Data(...)
# modified_data = remove_k_hop_subgraph(data, k=2)
