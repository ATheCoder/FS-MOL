from typing import List
from torch_geometric.utils import to_networkx
import math
from itertools import islice
from torch_geometric.data import Data
import torch
import networkx as nx
import operator


def nx_to_pyg(G, data):
    # Create a mapping of old node indices to new node indices
    node_mapping = {old: new for new, old in enumerate(G.nodes)}

    # Reconstruct edge_index with new node indices
    edge_index = (
        torch.tensor([(node_mapping[i], node_mapping[j]) for i, j in G.edges()]).t().contiguous()
    )

    edge_index = edge_index.int()

    # Reconstruct node features with new node indices
    x = data.x[list(node_mapping.keys())]
    x = x.int()

    # Reconstruct node positions with new node indices
    pos = data.pos[list(node_mapping.keys())]
    pos = pos.float()

    return Data(x=x, edge_index=edge_index, pos=pos)


def girvan_newman(G):
    """
    Applies the Girvan-Newman algorithm on the graph G
    and returns a generator of lists of nodes for each connected component
    after each step.
    """
    # The copy of G here is to avoid any side-effect on the original graph
    G = G.copy()
    while G.number_of_edges() > 0:
        # Compute edge betweenness centrality
        centrality = nx.edge_betweenness_centrality(G)
        # Identify the edge with maximum centrality
        max_centrality_edge = max(centrality.items(), key=operator.itemgetter(1))[0]
        # Remove the edge with highest centrality
        G.remove_edge(*max_centrality_edge)
        # Get the connected components and convert to list of nodes
        components = [list(c) for c in nx.connected_components(G)]
        yield components  # Each time we yield. We are yielding the complete graph.


def select_subgraph_by_indices(graph, node_indices):
    subgraph = nx.Graph()

    for index in node_indices:
        node = list(graph.nodes)[index]
        subgraph.add_node(node)

    for u, v in graph.edges:
        if u in node_indices and v in node_indices:
            subgraph.add_edge(u, v)

    return subgraph


def does_substructure_have_single_node(subgraphs):
    single_nodes = [g for g in subgraphs if len(g) == 1]

    return len(single_nodes) > 0


def get_unconnected_graph(G, data, unconnected_graph) -> List[Data]:
    subgraphs = []
    for subgraph_indices in unconnected_graph:
        selected_subgraph = select_subgraph_by_indices(G, subgraph_indices)
        subgraph_data = nx_to_pyg(selected_subgraph, data)
        subgraphs.append(subgraph_data)

    return subgraphs


def get_a_division(data: Data):
    G = to_networkx(data, to_undirected=True)
    node_count = data.x.shape[0]
    if node_count < 6:
        return [[data]]
    random_max = math.floor(node_count / 10) - 1
    if random_max <= 0:
        return [[data]]
    random_num = random_max
    # random_num = random.choice(list(range(random_max)))
    # print(f'Node Count: {node_count}, Random Number: {random_num}')
    # TODO: We can get the maximally unconnected graph.
    sccs = list(islice(girvan_newman(G), random_num + 1))
    sccs = [subgraphs for subgraphs in sccs if not does_substructure_have_single_node(subgraphs)]

    subgraph_count = len(sccs)

    if subgraph_count == 0:
        return [[data]]
    elif subgraph_count < random_num + 1:
        random_num = subgraph_count - 1

    return [get_unconnected_graph(G, data, unconnected_graph) for unconnected_graph in sccs]
