import torch
from torch import nn

class CrossAttention(nn.Module):
    def __init__(self, feat1_dim, feat2_dim, d_k, d_v, attn_drop = 0.):
        super().__init__()
        self.scale = d_k ** -0.5
        
        
        self.wq = nn.Linear(feat1_dim, d_k)
        self.wk = nn.Linear(feat2_dim, d_k)
        self.wv = nn.Linear(feat2_dim, d_v)
        
        self.attn_drop = nn.Dropout(attn_drop)
        
    def forward(self, feat1, feat2):
        q = self.wq(feat1) # Batch * d_k
        k = self.wk(feat2) # Batch * d_k
        v = self.wv(feat2) # Batch * d_v
        
        attn = (q @ k.transpose(-2, -1)) # Batch * Batch
        attn = attn.softmax(dim=-1) 
        attn = self.attn_drop(attn)
        
        result = (attn @ v) # Batch * Batch @ Batch * d_v
        
        return result # Batch * d_v


class CrossAttentionBlock(nn.Module):
    def __init__(self, feat1_dim, feat2_dim, d_k, d_v, d_ff, attn_drop = 0.2, n_heads = 1) -> None:
        super().__init__()
        
        self.cross_attention = nn.ModuleList(
            [
                CrossAttention(feat1_dim, feat2_dim, d_k=d_k, d_v=d_v, attn_drop=attn_drop) for _ in range(n_heads)
            ]
        )
        
        self.head_aggr = nn.Linear(n_heads * d_v, feat1_dim)
        
        # Layer Norms for both features
        self.feat1_normalizer = nn.LayerNorm(feat1_dim)
        self.feat2_normalizer = nn.LayerNorm(feat2_dim)
        
        self.fused_embedding_normalizer = nn.LayerNorm(feat1_dim)
        
        # Fused embedding FFN
        self.fused_norm_FFN = nn.Sequential(nn.Linear(feat1_dim, d_ff), nn.GELU(), nn.Linear(d_ff, feat1_dim))
        
        self.FFN_dropout = nn.Dropout(0.2)
    
    def forward(self, feat1, feat2):
        feat1_norm = self.feat1_normalizer(feat1)
        feat2_norm = self.feat2_normalizer(feat2)
        
        fused_embedding = torch.cat([attn_head(feat1_norm, feat2_norm) for attn_head in self.cross_attention], dim=-1)
        
        fused_embedding = self.head_aggr(fused_embedding)
        
        # Addition Step
        fused_embedding = feat1 + fused_embedding
        
        fused_embedding_norm = self.fused_embedding_normalizer(fused_embedding)
        
        fused_embedding_norm = self.fused_norm_FFN(fused_embedding_norm)
        
        fused_embedding_norm = self.FFN_dropout(fused_embedding_norm)
        
        return fused_embedding + fused_embedding_norm

class BidirectionalAttention(nn.Module):
    def __init__(self, graph_embedding_dim, mol_descs_dim, n_heads = 8, d_ff = 3072, output_dim = 768):
        super().__init__()
        
        d_k_graph_to_desc = d_v_graph_to_desc = graph_embedding_dim // n_heads
        self.graph_to_desc_attn = CrossAttentionBlock(graph_embedding_dim, mol_descs_dim, d_ff=d_ff, d_k=d_k_graph_to_desc, d_v=d_v_graph_to_desc, n_heads=n_heads)
        
        d_k_desct_to_graph = d_v_desc_to_graph = mol_descs_dim // n_heads
        self.desc_to_graph_attn = CrossAttentionBlock(mol_descs_dim, graph_embedding_dim, d_ff=d_ff, d_k=d_k_desct_to_graph, d_v=d_v_desc_to_graph, n_heads=n_heads)
        
        self.aggr_proj = nn.Sequential(
            nn.Linear(graph_embedding_dim + mol_descs_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, output_dim),
        )
        
    def forward(self, graph_embeddings, mol_descs):
        attn_graph_to_desc = self.graph_to_desc_attn(graph_embeddings, mol_descs)
        
        attn_desc_to_graph = self.desc_to_graph_attn(mol_descs, graph_embeddings)
        
        result = torch.cat([attn_graph_to_desc, attn_desc_to_graph], dim=-1)
        
        return self.aggr_proj(result)
    