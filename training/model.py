import timm
import torch.nn as nn


class SkinCancerModel(nn.Module):
    def __init__(self, model_name='resnext50_32x4d', pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, 9)

    def forward(self, x):
        x = self.model(x)
        return x