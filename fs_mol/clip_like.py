import pytorch_lightning as pl
from fs_mol.modules.graph_feature_extractor import GraphFeatureExtractor

from torch import nn
from torch.nn import functional as F
import torch

class FingerprintEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        
    def forward(self, x):
        return self.ffn(x)
    
class FingerprintEncoderWrapper(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        
        self.encoder = FingerprintEncoder(input_dim, hidden_dim, output_dim, dropout)
        
    def forward(self, datapoint):
        fingerprints = datapoint.fingerprint.reshape(-1, 2048)
        
        return self.encoder(fingerprints.float())
        

class ClipLike(pl.LightningModule):
    def __init__(self, temp, graph_encoder_config):
        super().__init__()
        self.temp
        self.graph_encoder = GraphFeatureExtractor(graph_encoder_config)
        
        self.fingerprint_encoder = FingerprintEncoder(2048, 1024, 512, 512)
    
    def training_step(self, batch, batch_idx):
        encoded_graphs = self.graph_encoder(batch)
        encoded_fingerprints = self.fingerprint_encoder(batch.fingerprint)
        
        logits = encoded_graphs @ encoded_fingerprints.T / self.temp
        
        graph_similarity = encoded_graphs @ encoded_fingerprints.T
        fingerprints_similarity = encoded_fingerprints @ encoded_graphs.T
        
        targets = F.softmax((graph_similarity + fingerprints_similarity) / 2 * self.temp, dim=-1)
        
        graphs_loss = F.cross_entropy(logits, targets, reduction="none")
        fingerprints_loss = F.cross_entropy(logits.T, targets.T, reduction="none")
        
        loss = graphs_loss + fingerprints_loss / 2.0
        
        return loss.mean()
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)