import torch
import torch.nn as nn
from models.cnn_model import CNNFeatureExtractor
from models.pose_estimation import PoseFeatureExtractor

class SignPoseFusionModel(nn.Module):
    def __init__(self, num_classes=10):
        super(SignPoseFusionModel, self).__init__()
        self.cnn = CNNFeatureExtractor()
        self.pose = PoseFeatureExtractor()

        self.classifier = nn.Sequential(
            nn.Linear(256 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, img, pose):
        img_feat = self.cnn(img)
        pose_feat = self.pose(pose)
        combined = torch.cat((img_feat, pose_feat), dim=1)
        out = self.classifier(combined)
        return out

if __name__ == "__main__":
    model = SignPoseFusionModel(num_classes=5)
    dummy_img = torch.randn(4, 3, 224, 224)
    dummy_pose = torch.randn(4, 42)
    out = model(dummy_img, dummy_pose)
    print("Final Output Shape:", out.shape)
