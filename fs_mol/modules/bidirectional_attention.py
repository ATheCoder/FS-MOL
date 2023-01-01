import torch
from torch import nn

class CrossAttention(nn.Module):
    def __init__(self, feat1_dim, feat2_dim, dim, qkv_bias = False, attn_drop = 0.2, proj_drop = 0.):
        super().__init__()
        self.scale = dim ** -0.5
        
        
        self.wq = nn.Linear(feat1_dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(feat2_dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(feat2_dim, dim, bias=qkv_bias)
        
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
        self.graph_to_desc_attn = CrossAttention(graph_embedding_dim, mol_descs_dim, dim)
        
        self.desc_to_graph_attn = CrossAttention(mol_descs_dim, graph_embedding_dim, dim)
        
        self.aggr_proj = nn.Linear(2 * dim, emb_dim)
        
    def forward(self, graph_embeddings, mol_descs):
        attn_graph_to_desc = self.graph_to_desc_attn(graph_embeddings, mol_descs)
        
        attn_desc_to_graph = self.desc_to_graph_attn(mol_descs, graph_embeddings)
        
        result = torch.cat([attn_graph_to_desc, attn_desc_to_graph], dim=-1)
        
        return self.aggr_proj(result)
    