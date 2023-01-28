import torch
import torch_geometric.utils as utils

def add_subgraph_info(graph, k_hops: int):
    num_nodes = graph.x.shape[0]
    
    node_indices = []
    edge_indices = []
    edge_attributes = []
    indicators = []
    edge_index_start = 0
    
    for node_idx in range(num_nodes):
        sub_nodes, sub_edge_index, _, edge_mask = utils.k_hop_subgraph(
                            node_idx, 
                            k_hops, 
                            graph.edge_index,
                            relabel_nodes=True, 
                            num_nodes=num_nodes
                            )

        node_indices.append(sub_nodes)
        edge_indices.append(sub_edge_index + edge_index_start)
        indicators.append(torch.zeros(sub_nodes.shape[0]).fill_(node_idx))
        edge_attributes.append(graph.edge_attr[edge_mask]) # CHECK THIS DIDN"T BREAK ANYTHING
        edge_index_start += len(sub_nodes)
        
    graph.subgraph_node_index = torch.cat(node_indices)
    graph.subgraph_edge_index = torch.cat(edge_indices, dim=1)
    graph.subgraph_indicator_index = torch.cat(indicators)
    graph.subgraph_edge_attr = torch.cat(edge_attributes)
    
    return graph