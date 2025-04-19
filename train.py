import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.fusion_model import SignFusionModel
from utils.dataset import SignLanguageDataset
from utils.metrics import accuracy_score
from config import Config

# Prepare dataset
dataset = SignLanguageDataset(Config.DATA_PATH, Config.SKELETON_PATH, Config.IMAGE_SIZE)
dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

# Initialize model
model = SignFusionModel(num_classes=Config.NUM_CLASSES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

# Training loop
for epoch in range(Config.EPOCHS):
    total_loss = 0
    all_preds = []
    all_labels = []

    model.train()
    for images, keypoints, labels in dataloader:
        images, keypoints, labels = images.to(device), keypoints.to(device), labels.to(device)

        outputs = model(images, keypoints)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_preds, all_labels)
    print(f"Epoch {epoch + 1}/{Config.EPOCHS}, Loss: {total_loss:.4f}, Accuracy: {acc:.4f}")

# Save model
torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)