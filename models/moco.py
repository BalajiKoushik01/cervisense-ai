import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class MoCov3(nn.Module):
    def __init__(self, encoder, dim=512, mlp_dim=4096, T=0.2):
        super().__init__()
        self.T = T
        self.encoder_q = encoder
        
        self.head_q = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.BatchNorm1d(mlp_dim), nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, mlp_dim), nn.BatchNorm1d(mlp_dim), nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, dim)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.BatchNorm1d(mlp_dim), nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, dim)
        )
        
        self.encoder_k = copy.deepcopy(encoder)
        self.head_k = copy.deepcopy(self.head_q)
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        for param in self.head_k.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def _momentum_update(self, m=0.996):
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data = p_k.data * m + p_q.data * (1.0 - m)
        for p_q, p_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            p_k.data = p_k.data * m + p_q.data * (1.0 - m)
    
    def forward(self, x1, x2):
        q1 = self.predictor(self.head_q(self.encoder_q(x1)))
        q2 = self.predictor(self.head_q(self.encoder_q(x2)))
        
        with torch.no_grad():
            self._momentum_update()
            k1 = self.head_k(self.encoder_k(x1))
            k2 = self.head_k(self.encoder_k(x2))
        
        loss = self._loss(q1, k2) + self._loss(q2, k1)
        return loss * 0.5
    
    def _loss(self, q, k):
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        labels = torch.arange(logits.shape[0], device=logits.device)
        return F.cross_entropy(logits, labels)
