import torch
from torch import nn

class CrossAttention(nn.Module):
    def __init__(self, feat1_dim, feat2_dim, dim, qkv_bias = False, attn_drop = 0., proj_drop = 0.):
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


class CrossAttentionBlock(nn.Module):
    def __init__(self, feat1_dim, feat2_dim, dim, qkv_bias = False, attn_drop = 0., proj_drop = 0.) -> None:
        super().__init__()
        
        self.cross_attention = CrossAttention(dim, dim, dim, qkv_bias, attn_drop, proj_drop)
        
        # These two are used to make the dim the same for both feat1 and feat2
        self.feat1_FFN = nn.Sequential(nn.Linear(feat1_dim, dim), nn.ReLU())
        self.feat2_FFN = nn.Sequential(nn.Linear(feat2_dim, dim), nn.ReLU())
        
        # Layer Norms for both features
        self.feat1_normalizer = nn.LayerNorm(dim)
        self.feat2_normalizer = nn.LayerNorm(dim)
        
        self.fused_embedding_normalizer = nn.LayerNorm(dim)
        
        # Fused embedding FFN
        self.fused_norm_FFN = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())
    
    def forward(self, feat1, feat2):
        feat1 = self.feat1_FFN(feat1)
        feat2 = self.feat2_FFN(feat2)
        
        feat1_norm = self.feat1_normalizer(feat1)
        feat2_norm = self.feat2_normalizer(feat2)
        
        fused_embedding = self.cross_attention(feat1_norm, feat2_norm)
        
        # Addition Step
        fused_embedding = feat1 + fused_embedding
        
        fused_embedding_norm = self.fused_embedding_normalizer(fused_embedding)
        
        fused_embedding_norm = self.fused_norm_FFN(fused_embedding_norm)
        
        return fused_embedding + fused_embedding_norm

class BidirectionalAttention(nn.Module):
    def __init__(self, graph_embedding_dim, mol_descs_dim, dim, emb_dim):
        super().__init__()
        self.graph_to_desc_attn = CrossAttentionBlock(graph_embedding_dim, mol_descs_dim, dim)
        
        self.desc_to_graph_attn = CrossAttentionBlock(mol_descs_dim, graph_embedding_dim, dim)
        
        self.aggr_proj = nn.Linear(2 * dim, emb_dim)
        
    def forward(self, graph_embeddings, mol_descs):
        attn_graph_to_desc = self.graph_to_desc_attn(graph_embeddings, mol_descs)
        
        attn_desc_to_graph = self.desc_to_graph_attn(mol_descs, graph_embeddings)
        
        result = torch.cat([attn_graph_to_desc, attn_desc_to_graph], dim=-1)
        
        return self.aggr_proj(result)
    