import torch
from torch_geometric.loader import DataLoader
from fs_mol.augmentation_transforms import SubGraphAugmentation
from fs_mol.data.fsmol_batcher import FSMolBatch, Feature_Extractor_FSMolBatch
from fs_mol.data.self_supervised_learning import FSMolSelfSupervisedInMemory
from fs_mol.modules.gnn import GNNBlock, GNNConfig, PyG_RelationalMP, RelationalMultiAggrMP
from fs_mol.modules.graph_feature_extractor import GraphFeatureExtractor, GraphFeatureExtractorConfig
from fs_mol.modules.pyg_gnn import PyG_GNNBlock, PyG_GraphFeatureExtractor
from torch_geometric.data import Data

# Generate a batch of Graphs

dataset_subgraph = FSMolSelfSupervisedInMemory('./datasets/self-supervised')
batch_size = 16

dl = DataLoader(dataset_subgraph, batch_size=batch_size)
example_batch = next(iter(dl))

# Convert that batch of Graphs into a representation that Feature Extractor can use

def convert_batch_to_legacy(batch: Data):
    num_graphs = batch.y.shape[0]
    
    node_features = batch.x
    
    t_edge_index = batch.edge_index.t()
    
    adjacency_lists = [[], [], []]
    
    for i, edge in enumerate(t_edge_index):
        edge_type = batch.edge_attr[i]
        
        adjacency_lists[int(edge_type)].append(edge)    
    
    for i in range(3):
        if len(adjacency_lists[i]) < 1:
            adjacency_lists[i] = torch.empty((0, 2), dtype=torch.int64)
            continue
        adjacency_lists[i] = torch.vstack(adjacency_lists[i])
        
    
    node_to_graph = batch.batch
    
    return Feature_Extractor_FSMolBatch(num_graphs=num_graphs, num_nodes=node_features.shape[0], num_edges=t_edge_index.shape[0], node_features=node_features, adjacency_lists=adjacency_lists, node_to_graph=node_to_graph)
        
c = convert_batch_to_legacy(example_batch)

print(c)


def generate_example_molecule_graph():
    x = torch.rand((10, 128))
    
    adj_lists = [torch.tensor([[0, 1], [1, 0]]), torch.tensor([[2, 3], [3, 2]]), torch.tensor([[5, 4], [4, 5]])]
    
    edge_index = torch.tensor([[0, 1, 2, 3, 5, 4], [1, 0, 3, 2, 4, 5]], dtype=torch.long)
    
    edge_attr = torch.tensor([0, 0, 1, 1, 2, 2])

    return x, adj_lists, Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


graph_feature_extractor_config = GraphFeatureExtractorConfig()

gnn_config = GNNConfig()

def getParameterCount(model):
    return sum([x.numel() for x in model.parameters()])

x, adj_lists, pyg_data = generate_example_molecule_graph()

def compare_models(model1, model2, data1, data2):
    (x, adj_lists) = data1
    
    for p in model1.parameters():
        p.data.fill_(0.05)
    
    for p in model2.parameters():
        p.data.fill_(0.05)
    
    
    output_model1 = model1(node_representations=x, adj_lists=adj_lists)
    
    output_model2 = model2(x=data2.x, edge_index=data2.edge_index, edge_attr=data2.edge_attr)
    # Should have the same number of Parameters:
    assert getParameterCount(model1) == getParameterCount(model2)
    # print(output_model1[0])
    # print('------')
    # print(output_model2[0])
    
    # Outputs should be the same:
    assert torch.allclose(output_model2, output_model1)
    

relational_mp_config = {
    "hidden_dim": gnn_config.hidden_dim // gnn_config.num_heads,
    "msg_dim": gnn_config.per_head_dim,
    "num_edge_types": gnn_config.num_edge_types,
    "message_function_depth": gnn_config.message_function_depth,
    "use_pna_scalers":True
}

compare_models(GNNBlock(GNNConfig()), PyG_GNNBlock(GNNConfig()), (x, adj_lists), pyg_data)



