import torch
import torch.nn as nn
import torchvision.models as models

class CNNFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(CNNFeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)
        modules = list(resnet.children())[:-1]  # Remove the classification layer
        self.feature_extractor = nn.Sequential(*modules)
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = CNNFeatureExtractor()
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print("CNN Output Shape:", output.shape)
