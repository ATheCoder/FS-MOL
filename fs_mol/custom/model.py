import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.data import Data
from torch_scatter import scatter

class SimpleGraphConv(torch.nn.Module):
    def __init__(self, depth = 10):
        super().__init__()
        self.depth = depth

        self.embedding = Linear(32, 128, bias=False)
        self.graph_convs = []

        self.weights_for_sum = Linear(depth * 128, 1)

        for _ in range(self.depth):
            self.graph_convs.append(GraphConv(128, 128))
        
        self.sum_to_graph = Linear(depth * 128, 32)


    def forward(self, graph: Data, batch_map = None):

        h_0 = self.embedding(graph.x)

        node_representations = []
        layer_out = h_0

        for i in range(self.depth):
            layer_out = self.graph_convs[i](layer_out, graph.edge_index)
            node_representations.append(layer_out)

        out = torch.cat(node_representations, dim=1)
        sum_weights = F.softmax(self.weights_for_sum(out), dim=-1)
        out = (sum_weights * out)

        if batch_map != None:
            sum_out = scatter(out, batch_map, dim=0)
        
        else:
            sum_out = out.sum(dim=0)


        return self.sum_to_graph(sum_out)
