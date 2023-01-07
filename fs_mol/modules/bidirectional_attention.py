import torch
from torch import nn

class CrossAttention(nn.Module):
    def __init__(self, feat1_dim, feat2_dim, dim, attn_drop = 0.):
        super().__init__()
        self.scale = dim ** -0.5
        
        
        self.wq = nn.Linear(feat1_dim, dim)
        self.wk = nn.Linear(feat2_dim, dim)
        self.wv = nn.Linear(feat2_dim, dim)
        
        self.attn_drop = nn.Dropout(attn_drop)
        
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
    def __init__(self, feat1_dim, feat2_dim, attn_drop = 0.) -> None:
        super().__init__()
        
        self.cross_attention = CrossAttention(feat1_dim, feat2_dim, feat1_dim, attn_drop)
        
        # Layer Norms for both features
        self.feat1_normalizer = nn.LayerNorm(feat1_dim)
        self.feat2_normalizer = nn.LayerNorm(feat2_dim)
        
        self.fused_embedding_normalizer = nn.LayerNorm(feat1_dim)
        
        # Fused embedding FFN
        self.fused_norm_FFN = nn.Sequential(nn.Linear(feat1_dim, 2048), nn.ReLU(), nn.Linear(2048, feat1_dim))
    
    def forward(self, feat1, feat2):
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
        self.graph_to_desc_attn = CrossAttentionBlock(graph_embedding_dim, mol_descs_dim)
        
        self.desc_to_graph_attn = CrossAttentionBlock(mol_descs_dim, graph_embedding_dim)
        
        self.aggr_proj = nn.Sequential(
            nn.Linear(graph_embedding_dim + mol_descs_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, emb_dim),
        )
        
    def forward(self, graph_embeddings, mol_descs):
        attn_graph_to_desc = self.graph_to_desc_attn(graph_embeddings, mol_descs)
        
        attn_desc_to_graph = self.desc_to_graph_attn(mol_descs, graph_embeddings)
        
        result = torch.cat([attn_graph_to_desc, attn_desc_to_graph], dim=-1)
        
        return self.aggr_proj(result)
    