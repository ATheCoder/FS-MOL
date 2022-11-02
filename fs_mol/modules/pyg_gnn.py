from typing import Optional
import torch
from torch.nn import Embedding, Linear, Module
from torch_geometric.data import Data
from torch_geometric.nn import GINConv
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter

from fs_mol.modules.gnn import BOOMLayer, GNNBlock, GNNConfig, PyG_RelationalMP, RelationalMultiAggrMP
from fs_mol.modules.graph_feature_extractor import GraphFeatureExtractorConfig
from fs_mol.modules.graph_readout import make_readout_model

# TODO: Add BatchNormalization (Hint: model.py in GraphCL)
# TODO: How do we aggregate the hidden representations of a graph?
#           Should we aggregate via what is happening in the FS-Mol paper or what is happening inside GIN?

# FS-MOL's prototypical network uses features that are not dependant on the molecular Graph. Should we test whether training a model that doesn't use these features still achieves a good performance?

# How are graph representations calculated using FS-MOL's prototypical network?
# - 1. **GraphFeatureExtractor**.
#     - 1. Linear Layer without bias (Embedding Layer?) [Done]
#       2. Node Representations via a **GNN Module** (Multiple **GNNBlock**s)
#       -   1. (Optional: Default True) make the edges bidirectional. [Done]
#           2. An array of node (Default count: 8) representations that are outputed via a number of **GNNBlock**s
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

SMALL_NUMBER = 1e-7

class PyG_GNNBlock(Module):
    def __init__(self, config: GNNConfig) -> None:
        super().__init__()
        
        self.config = config
        
        if config.use_rezero_scaling:
            self.alpha = torch.nn.Parameter(torch.full(size=(1,), fill_value=SMALL_NUMBER))
            
        self.mp_layer_in_dim = config.hidden_dim // config.num_heads
        
        # This is the PNA Implementation
        self.mp_layers = torch.nn.ModuleList([PyG_RelationalMP(
                            hidden_dim=self.mp_layer_in_dim,
                            msg_dim=config.per_head_dim,
                            num_edge_types=config.num_edge_types,
                            message_function_depth=config.message_function_depth,
                            use_pna_scalers=True
                            ) for _ in range(config.num_heads)])
        
        
        
        total_msg_dim = sum(mp_layer.message_size for mp_layer in self.mp_layers)
        self.msg_out_projection = Linear(in_features=total_msg_dim, out_features=config.hidden_dim)
        self.dropout_layer = torch.nn.Dropout(p=config.dropout_rate)
        if config.intermediate_dim > 0:
            self.boom_layer: Optional[BOOMLayer] = BOOMLayer(
                inout_dim=config.hidden_dim,
                intermediate_dim=config.intermediate_dim,
                dropout=config.dropout_rate,
            )
            self.boom_norm_layer: Optional[torch.nn.Module] = torch.nn.LayerNorm(
                normalized_shape=config.hidden_dim
            )
        else:
            self.boom_layer = None
            self.boom_norm_layer = None
    
    def forward(self, x, edge_index, edge_attr):
        node_representations = x
        aggregated_messages = []
        for i, mp_layer in enumerate(self.mp_layers):
            sliced_node_representations = node_representations[
                :, i * self.mp_layer_in_dim : (i + 1) * self.mp_layer_in_dim
            ]
            aggregated_messages.append(mp_layer(x=sliced_node_representations, edge_index=edge_index, edge_attr=edge_attr))
        
        new_representations = self.msg_out_projection(torch.cat(aggregated_messages, dim=-1))
        new_representations = self.dropout_layer(new_representations)
        
        if self.config.use_rezero_scaling:
            new_representations = self.alpha * new_representations
        
        node_representations = node_representations + new_representations
        
        if self.boom_layer is not None and self.boom_norm_layer is not None:
            boomed_representations = self.dropout_layer(
                self.boom_layer(self.boom_norm_layer(node_representations))
            )
            if self.config.use_rezero_scaling:
                boomed_representations = self.alpha * boomed_representations
            node_representations = node_representations + boomed_representations
        return node_representations
            

class PyG_GraphFeatureExtractor(Module):
    def __init__(self, config: GraphFeatureExtractorConfig) -> None:
        super().__init__()
        
        self.config = config
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Input dimension is 32
        self.embedding_layer = Linear(config.initial_node_feature_dim, config.gnn_config.hidden_dim, bias=False)
        self.layers = torch.nn.ModuleList([PyG_GNNBlock(GNNConfig()) for _ in range(config.gnn_config.num_layers)])
        
        if config.readout_config.use_all_states:
            readout_node_dim = (config.gnn_config.num_layers + 1) * config.gnn_config.hidden_dim
        else:
            readout_node_dim = config.gnn_config.hidden_dim
        
        self.readout = make_readout_model(
            self.config.readout_config,
            readout_node_dim,
        )
        
        if self.config.output_norm == "off":
            self.final_norm_layer: Optional[torch.nn.Module] = None
        elif self.config.output_norm == "layer":
            self.final_norm_layer = torch.nn.LayerNorm(
                normalized_shape=self.config.readout_config.output_dim
            )
        elif self.config.output_norm == "batch":
            self.final_norm_layer = torch.nn.BatchNorm1d(
                num_features=self.config.readout_config.output_dim
            )
        
    def forward(self, graph: Data):
        x = self.embedding_layer(graph.x)
        edge_index = graph.edge_index
        # Make graph bidirectional:
        flipped_edge_index = torch.flip(graph.edge_index, dims=(0,))
        edge_index = torch.cat([graph.edge_index, flipped_edge_index], dim=1)
        edge_attr = graph.edge_attr.tile((2,))
        
        all_node_representations = [x]
        last_conv = x
        
        for i in range(len(self.layers)):
            last_conv = self.layers[i](last_conv, edge_index=edge_index, edge_attr=edge_attr)
            all_node_representations.append(last_conv)

        if self.config.readout_config.use_all_states:
            readout_node_reprs = torch.cat(all_node_representations, dim=-1)
        else:
            readout_node_reprs = all_node_representations[-1]
        
        
        mol_representations = self.readout(
            node_embeddings=readout_node_reprs,
            node_to_graph_id=graph.batch,
            num_graphs=graph.y.shape[0],
        )

        if self.final_norm_layer is not None:
            mol_representations = self.final_norm_layer(mol_representations)

        return mol_representations
        
        