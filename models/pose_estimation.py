import torch
import torch.nn as nn

class PoseFeatureExtractor(nn.Module):
    def __init__(self, input_dim=42):  # 21 keypoints * (x,y)
        super(PoseFeatureExtractor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    model = PoseFeatureExtractor()
    dummy_input = torch.randn(4, 42)
    output = model(dummy_input)
    print("Pose Output Shape:", output.shape)
