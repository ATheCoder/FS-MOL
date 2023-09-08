from typing import List
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.utils import unbatch


def generate_batch_index(lengths: Tensor):
    return torch.repeat_interleave(torch.arange(len(lengths), device=lengths.device), lengths)


def generate_mask(lengths: Tensor):
    max_len = lengths.max()

    mask = torch.arange(max_len.item(), device=lengths.device).unsqueeze(0)
    return mask < lengths.unsqueeze(1)


def batch_variable_length(list_of_sequences: List[Tensor]):
    batch = pad_sequence(list_of_sequences, True)
    lengths = torch.tensor(
        [t.shape[0] for t in list_of_sequences], device=list_of_sequences[0].device
    )

    return batch, lengths


def separate_qsl(graph_reprs, labels, is_query, batch_index):
    # Support Vectors Positive, Support Labels
    bool_is_query = is_query.bool()

    support_graphs = graph_reprs[~bool_is_query]
    support_labels = labels[~bool_is_query]
    support_batch_index = batch_index[~bool_is_query]

    support_graphs = unbatch(support_graphs, support_batch_index)
    support_labels = unbatch(support_labels, support_batch_index)
    batch_support_graphs, support_graph_lengths = batch_variable_length(support_graphs)
    batch_support_labels, _ = batch_variable_length(support_labels)

    # Query Vectors, Query Labels
    query_graphs = graph_reprs[bool_is_query]
    query_labels = labels[bool_is_query]
    query_batch_index = batch_index[bool_is_query]

    query_graphs = unbatch(query_graphs, query_batch_index)
    query_labels = unbatch(query_labels, query_batch_index)

    batch_query_graphs, query_graph_lengths = batch_variable_length(query_graphs)
    batch_query_labels, _ = batch_variable_length(query_labels)

    return (
        batch_support_graphs,
        batch_support_labels,
        support_graph_lengths,
        batch_query_graphs,
        batch_query_labels,
        query_graph_lengths,
    )


def batch_2d_tensor(tensor: Tensor, lengths: Tensor):
    """
    Converts a 2D Tensor representing a collection of vectors to a padded 3D Tensor representing batched collection of vectors.
    Parameters:
    tensor -- The 2D tensor.
    lengths -- The a list representing the number of vectors on each batch
    """
    list_of_sequences = torch.split(tensor, lengths.tolist())
    res, lens = batch_variable_length(list(list_of_sequences))
    return res
