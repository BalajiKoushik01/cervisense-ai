import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

class CerviSenseEncoder(nn.Module):
    def __init__(self, pretrained=True, embedding_dim=512):
        super().__init__()
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        base = efficientnet_v2_s(weights=weights)
        
        self.features = base.features
        self.avgpool = base.avgpool
        self.embed_dim = 1280 
        
        self.projector = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, embedding_dim),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        embedding = self.projector(x)
        return embedding
    
    def get_features(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)
