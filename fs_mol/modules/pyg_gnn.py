import torch
from torch.nn import Embedding, Linear, Module
from torch_geometric.data import Data
from torch_geometric.nn import GINConv
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter

# TODO: Add BatchNormalization (Hint: model.py in GraphCL)
# TODO: How do we aggregate the hidden representations of a graph?
#           Should we aggregate via what is happening in the FS-Mol paper or what is happening inside GIN?

# FS-MOL's prototypical network uses features that are not dependant on the molecular Graph. Should we test whether training a model that doesn't use these features still achieves a good performance?

# How are graph representations calculated using FS-MOL's prototypical network?
# - 1. **GraphFeatureExtractor**.
#     - 1. Linear Layer without bias (Embedding Layer?) [Done]
#       2. Node Representations via a **GNN Module** (Multiple **GNNBlock**s)
#       -   1. (Optional: Default True) make the edges bidirectional.
#           2. An array of node representations that are outputed via a number of **GNNBlock**s
#           -   1. Float Tensor of shape (num_nodes, config.hidden_dim)
#               2. mp_layers can be one of the following:
#               -   1. Single **MultiHeadAttention**
#                   2. Num of Heads of **MultiAggr**
#                   3. Num of Heads of **PNA**
#                   4. Num of Heads of **Plain**
#               3. Each Node Representation Vector is sliced into equal numbers (depending on the number of Heads) and then the result is concatenated into an array. cat with dim -1
#               4. The resulting concatenated tensor is then passed into a *Linear Layer* with bias.
#               5. (Optional) use_rezero_scaling
#               6. The resulting tensor is called new_representations and *Added* to the begining node vector (Yes, the new_representations have an equal dimension to the begining vector)
#               7. (Optional) BoomLayer and add to representation again.
#               8. Float Tensor of shape (num_graphs, config.hidden_dim)
#       3. **make_readout_model**
#       -   1. One of the following: **CombinedGraphReadout**, **MultiHeadWeightedGraphReadout**, **UnweightedGraphReadout** (Default is: Combined)
#       -   2. 
#       4. (Optional Norm Layer) can be one of these |Off|LayernNorm|BatchNorm1d|
#       
#   2. Concat the graph features with all the other features (i.e: molecular weights, ...).
#   3. Mahalanobis distance and prototypical network training ...

class GeometricGNN(Module):
    def __init__(self, layer_count) -> None:
        super().__init__()
        
        # Input dimension is 32
        self.embedding_layer = Linear(32, 128, bias=False)
        
        self.layers = torch.nn.ModuleList([GINConv(Linear(128, 128)) for _ in range(layer_count)])
        
    def forward(self, graph: Data):
        x = self.embedding_layer(graph.x)
        # Make the Graph Bidirectional
        edge_index, _ = add_self_loops(graph.edge_index, num_nodes=x.size(0))
        
        features = [x]
        last_conv = x
        
        for i in range(len(self.layers)):
            last_conv = self.layers[i](last_conv, edge_index=edge_index)
            
            features.append(last_conv)
        
        node_representations = torch.cat(features, dim=-1)
        graph_representations = scatter(src=node_representations, index=graph.batch, dim=0, reduce='sum')
        
        return graph_representations
        
        