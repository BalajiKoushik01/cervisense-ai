import torch.nn as nn

class CerviSenseClassifier(nn.Module):
    def __init__(self, encoder, num_classes=5, dropout=0.4):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)
