import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

def compute_accuracy(y_pred, y_true):
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y_true).sum().item()
    return correct / len(y_true)

def evaluate_model(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, poses, labels in dataloader:
            images, poses, labels = images.to(device), poses.to(device), labels.to(device)
            outputs = model(images, poses)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred)

    return acc, report, matrix

def print_metrics(acc, report, matrix):
    print(f"Accuracy: {acc*100:.2f}%\n")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", matrix)
