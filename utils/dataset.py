import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SignLanguageDataset(Dataset):
    def __init__(self, image_dir, skeleton_path, image_size):
        self.image_dir = image_dir
        self.image_size = image_size
        self.images = sorted(os.listdir(image_dir))
        skeleton_data = np.load(skeleton_path)
        self.keypoints = skeleton_data['keypoints']
        self.labels = skeleton_data['labels']

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)

        keypoint = torch.tensor(self.keypoints[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, keypoint, label