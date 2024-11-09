import random
import torch
import networkx as nx
import random
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx

def remove_random_subgraph(graph, num_nodes_to_remove):
    # Create a copy of the input graph
    result_graph = graph.copy()

    # Compute the degree centrality of each node in the graph
    degree_centrality = nx.degree_centrality(graph)

    # Shuffle the nodes randomly
    nodes = list(graph.nodes())
    random.shuffle(nodes)

    # Sort shuffled nodes by degree centrality in ascending order
    # Nodes with lower degree centrality are more likely to be on the outskirts of the graph
    sorted_nodes = sorted(nodes, key=lambda node: degree_centrality[node])

    # Select nodes on the outskirts of the graph
    outskirts_nodes = sorted_nodes[:num_nodes_to_remove]

    # Remove the outskirts nodes and their edges from the copy
    result_graph.remove_nodes_from(outskirts_nodes)

    # Check if the resulting graph is fully connected
    if not nx.is_connected(result_graph):
        # Find the largest connected component in the resulting graph
        largest_component = max(nx.connected_components(result_graph), key=len)

        # Create a new graph containing only the largest connected component
        result_graph = result_graph.subgraph(largest_component).copy()

    return result_graph

def remove_based_on_distance(data, num_nodes):
    # Convert PyG data to NetworkX graph
    G = to_networkx(data, to_undirected=True)

    # Get the list of biconnected components
    G = remove_random_subgraph(G, num_nodes)
    # print(f"Removed biconnected component with {len(component_to_remove)} nodes. Remaining graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Convert the graph back to PyG format
    data_new = nx_to_pyg(G, data)

    return data_new

def nx_to_pyg(G, data):
    # Create a mapping of old node indices to new node indices
    node_mapping = {old: new for new, old in enumerate(G.nodes)}

    # Reconstruct edge_index with new node indices
    edge_index = torch.tensor([(node_mapping[i], node_mapping[j]) for i, j in G.edges()]).t().contiguous()

    # Reconstruct node features with new node indices
    x = data.x[list(node_mapping.keys())]

    # Reconstruct node positions with new node indices
    pos = data.pos[list(node_mapping.keys())]

    return Data(x=x, edge_index=edge_index, pos=pos)

def harmonic_series(n):
    result = 0.0
    for i in range(1, n+1):
        result += 1.0 / i
    return result

def generate_aug_n_times(data):
    node_num = data.x.shape[0]
    aug_ratio = int(node_num * 0.3)
    times = int(aug_ratio * harmonic_series(aug_ratio))
    return [(remove_based_on_distance(data, aug_ratio), data) for _ in range(times)]