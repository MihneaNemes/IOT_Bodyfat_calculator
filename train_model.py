# train_model.py
import torch
from torch import nn, optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image
import numpy as np
import os

# Load dataset
ds = load_dataset("TrainingDataPro/body-measurements-dataset")

# Dataset class
class BodyFatDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset["train"])

    def __getitem__(self, idx):
        data_point = self.dataset["train"][idx]
        image = data_point["image"]
        if not isinstance(image, Image.Image):
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image).convert("RGB")
        label = data_point["label"]
        if isinstance(label, dict) and "body_fat_percentage" in label:
            body_fat_percentage = label["body_fat_percentage"]
        elif isinstance(label, (int, float)):
            body_fat_percentage = label
        else:
            body_fat_percentage = 0.0
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(body_fat_percentage, dtype=torch.float32)

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# DataLoader
train_dataset = BodyFatDataset(ds, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Model setup
from torchvision.models import ResNet50_Weights
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training
num_epochs = 10
model.train()
for epoch in range(num_epochs):
    total_loss = 0.0
    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (i + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

# Save model weights
torch.save(model.state_dict(), "bodyfat_regressor.pth")
print("Model training complete and saved as 'bodyfat_regressor.pth'")
