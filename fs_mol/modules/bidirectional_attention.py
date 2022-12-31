import torch
from torch import nn

class CrossAttention(nn.Module):
    def __init__(self, dim, qkv_bias = False, attn_drop = 0., proj_drop = 0.):
        super().__init__()
        self.scale = dim ** -0.5
        
        
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        
        self.proj = nn.Linear(dim, dim)
        
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, feat1, feat2):
        q = self.wq(feat1)
        k = self.wk(feat2)
        v = self.wv(feat2)
        
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        result = (attn @ v)
        
        return result
        

class BidirectionalAttention(nn.Module):
    def __init__(self, graph_embedding_dim, mol_descs_dim, dim, emb_dim):
        super().__init__()
        
        self.graph_embedding_projector = nn.Sequential(nn.Linear(graph_embedding_dim, dim), nn.ReLU())
        
        self.molecular_desc_projector =  nn.Sequential(nn.Linear(mol_descs_dim, dim), nn.ReLU())
        
        self.graph_to_desc_attn = CrossAttention(dim)
        
        self.desc_to_graph_attn = CrossAttention(dim)
        
        self.aggr_proj = nn.Linear(2 * dim, emb_dim)
        
    def forward(self, graph_embeddings, mol_descs):
        graph_embeddings = self.graph_embedding_projector(graph_embeddings)
        
        mol_descs = self.molecular_desc_projector(mol_descs)
        
        attn_graph_to_desc = self.graph_to_desc_attn(graph_embeddings, mol_descs)
        
        attn_desc_to_graph = self.desc_to_graph_attn(mol_descs, graph_embeddings)
        
        result = torch.cat([attn_graph_to_desc, attn_desc_to_graph], dim=-1)
        
        return self.aggr_proj(result)
    
    
model = BidirectionalAttention(512, 200, 256, 512)

example_graph_embeddings = torch.rand((128, 512))
example_mol_descs = torch.rand((128, 200))


model(example_graph_embeddings, example_mol_descs)