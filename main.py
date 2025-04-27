import torch
from torch import nn, optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Load the Body Measurements Dataset
ds = load_dataset("TrainingDataPro/body-measurements-dataset")

# Print dataset structure to understand its format
print("Dataset structure example:")
print(f"Keys: {ds['train'][0].keys() if len(ds['train']) > 0 else 'Empty dataset'}")
print(f"Image type: {type(ds['train'][0]['image'])}")
print(f"Label example: {ds['train'][0]['label']}")


# Preprocess the images and body measurements
class BodyFatDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset["train"])

    def __getitem__(self, idx):
        data_point = self.dataset["train"][idx]

        # Get the image data
        image = data_point["image"]
        # Convert to PIL Image if it's not already
        if not isinstance(image, Image.Image):
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image).convert("RGB")

        # Get label (assuming it contains body fat percentage or can be processed to get it)
        label = data_point["label"]

        # Extract body fat percentage from label
        # This might need adjustment based on what's actually in the label
        if isinstance(label, dict) and "body_fat_percentage" in label:
            body_fat_percentage = label["body_fat_percentage"]
        elif isinstance(label, (int, float)):
            # If label is directly a number, use it as is
            body_fat_percentage = label
        else:
            # Default value if we can't determine body fat percentage
            body_fat_percentage = 0.0
            print(f"Warning: Could not extract body fat percentage from label: {label}")

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(body_fat_percentage, dtype=torch.float32)


# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Try printing more details about the dataset structure before proceeding
try:
    # Show a sample image to understand what we're working with
    sample_image = ds["train"][0]["image"]
    if isinstance(sample_image, np.ndarray):
        plt.imshow(sample_image)
        plt.title("Sample Image")
        plt.axis('off')
        plt.show()
    else:
        print(f"Image is not a numpy array, it's a {type(sample_image)}")

    # Investigate label structure
    sample_label = ds["train"][0]["label"]
    print(f"Label type: {type(sample_label)}")
    if isinstance(sample_label, dict):
        print(f"Label keys: {sample_label.keys()}")
except Exception as e:
    print(f"Error analyzing dataset: {e}")

# Load the dataset into a PyTorch DataLoader
try:
    train_dataset = BodyFatDataset(ds, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Test if we can get an item from the dataset
    test_image, test_label = train_dataset[0]
    print(f"Successfully loaded test item: image shape={test_image.shape}, label={test_label}")
except Exception as e:
    print(f"Error setting up dataset: {e}")
    raise e

# Define the model (ResNet for regression)
from torchvision import models
from torchvision.models import ResNet50_Weights

# Use weights argument instead of 'pretrained'
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 1)  # Modify the final fully connected layer for regression

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()  # For regression, we use Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 10
model.train()  # Set model to training mode

try:
    for epoch in range(num_epochs):
        total_loss = 0.0
        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)  # Add a dimension for regression output

            optimizer.zero_grad()  # Zero out the gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, targets)  # Calculate loss

            loss.backward()  # Backpropagate the gradients
            optimizer.step()  # Update the weights

            total_loss += loss.item()

            # Print progress every 5 batches
            if (i + 1) % 5 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "bodyfat_regressor.pth")
    print("Model saved!")
except Exception as e:
    print(f"Error during training: {e}")


# Optional: Evaluate on a few examples
def predict_body_fat(image_path, model):
    """Make a prediction using the trained model"""
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension

    # Make prediction
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor)

    return prediction.item()

# Example usage of prediction function (uncomment when you have an image to test)
test_image_path = r"C:\Users\mihne\OneDrive\Desktop\sarpili\IOT\pictures\testimg1.png"
predicted_body_fat = predict_body_fat(test_image_path, model)
print(f"Predicted Body Fat Percentage: {predicted_body_fat:.2f}%")