import torch
from fs_mol.modules.gnn import GNNBlock, GNNConfig, PyG_RelationalMP, RelationalMultiAggrMP
from fs_mol.modules.graph_feature_extractor import GraphFeatureExtractor, GraphFeatureExtractorConfig
from fs_mol.modules.pyg_gnn import PyG_GNNBlock, PyG_GraphFeatureExtractor
from torch_geometric.data import Data


def generate_example_molecule_graph():
    x = torch.rand((10, 32))
    
    adj_lists = [torch.tensor([[0, 1], [1, 0]]), torch.tensor([[2, 3], [3, 2]]), torch.tensor([[5, 4], [4, 5]])]
    
    edge_index = torch.tensor([[0, 1, 2, 3, 5, 4], [1, 0, 3, 2, 4, 5]])
    
    edge_attr = torch.tensor([0, 0, 1, 1, 2, 2])

    return x, adj_lists, Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


graph_feature_extractor_config = GraphFeatureExtractorConfig()

gnn_config = GNNConfig()

def getParameterCount(model):
    return sum([x.numel() for x in model.parameters()])


def compare_models(model1, model2):    
    x, adj_lists, pyg_data = generate_example_molecule_graph()
    
    for p in model1.parameters():
        p.data.fill_(0.5)
    
    for p in model2.parameters():
        p.data.fill_(0.5)
    
    output_model1 = model1(x=pyg_data.x, edge_index=pyg_data.edge_index, edge_attr=pyg_data.edge_attr)
    
    output_model2 = model2(x, adj_lists)
    
    # Should have the same number of Parameters:
    assert getParameterCount(model1) == getParameterCount(model2)
    
    # Outputs should be the same:
    assert torch.allclose(output_model2, output_model1)
    

relational_mp_config = {
    "hidden_dim": gnn_config.hidden_dim // gnn_config.num_heads,
    "msg_dim": gnn_config.per_head_dim,
    "num_edge_types": gnn_config.num_edge_types,
    "message_function_depth": gnn_config.message_function_depth,
    "use_pna_scalers":True
}

compare_models(PyG_RelationalMP(**relational_mp_config), RelationalMultiAggrMP(**relational_mp_config))
