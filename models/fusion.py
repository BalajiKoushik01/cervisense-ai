import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    def __init__(self, dim=1280, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, query_feat, key_value_feat):
        q = query_feat.unsqueeze(1)
        kv = key_value_feat.unsqueeze(1)
        attn_out, attn_weights = self.attn(q, kv, kv)
        return self.norm(attn_out.squeeze(1) + query_feat), attn_weights

class ModalityGate(nn.Module):
    def __init__(self, dim=1280):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 2),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, feat_a, feat_b, mask_a=None, mask_b=None):
        combined = torch.cat([feat_a, feat_b], dim=-1)
        weights = self.gate(combined)
        if mask_a is not None:
            weights[:, 0] = weights[:, 0] * mask_a
        if mask_b is not None:
            weights[:, 1] = weights[:, 1] * mask_b
        weights = F.normalize(weights, p=1, dim=-1)
        fused = weights[:, 0:1] * feat_a + weights[:, 1:2] * feat_b
        return fused, weights

class HCMAF(nn.Module):
    def __init__(self, encoder_colpo, encoder_histo, num_classes=5, dim=1280):
        super().__init__()
        self.encoder_colpo = encoder_colpo
        self.encoder_histo = encoder_histo
        
        self.self_attn_colpo = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.self_attn_histo = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        
        self.cross_attn_colpo2histo = CrossModalAttention(dim, num_heads=8)
        self.cross_attn_histo2colpo = CrossModalAttention(dim, num_heads=8)
        
        self.gate = ModalityGate(dim)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x_colpo=None, x_histo=None):
        has_colpo = x_colpo is not None
        has_histo = x_histo is not None
        
        if has_colpo:
            f_colpo = self.encoder_colpo.get_features(x_colpo)
        if has_histo:
            f_histo = self.encoder_histo.get_features(x_histo)
        
        if has_colpo and has_histo:
            f_c, _ = self.self_attn_colpo(
                f_colpo.unsqueeze(1), f_colpo.unsqueeze(1), f_colpo.unsqueeze(1))
            f_h, _ = self.self_attn_histo(
                f_histo.unsqueeze(1), f_histo.unsqueeze(1), f_histo.unsqueeze(1))
            f_c = f_c.squeeze(1)
            f_h = f_h.squeeze(1)
            
            f_c_enriched, cross_attn_c = self.cross_attn_colpo2histo(f_c, f_h)
            f_h_enriched, cross_attn_h = self.cross_attn_histo2colpo(f_h, f_c)
            
            fused, gate_weights = self.gate(f_c_enriched, f_h_enriched)
        
        elif has_colpo:
            fused, gate_weights = f_colpo, None
        else:
            fused, gate_weights = f_histo, None
        
        return self.classifier(fused), gate_weights
