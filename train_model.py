import os
import pandas as pd
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import models, transforms
import matplotlib.pyplot as plt


# Custom Dataset Class
class BodyMDataset(Dataset):
    def __init__(self, base_dir, subset_folder, transform=None):
        self.base_dir = base_dir
        self.subset_folder = subset_folder

        # Path to the measurements file for this subset
        measurements_file = os.path.join(base_dir, subset_folder, "measurements.csv")
        self.measurements = pd.read_csv(measurements_file)

        # Path to the folder containing mask images
        self.mask_dir = os.path.join(base_dir, subset_folder, "mask")

        # Optional mapping file
        self.subject_photo_map_file = os.path.join(base_dir, subset_folder, "subject_to_photo_map.csv")
        if os.path.exists(self.subject_photo_map_file):
            self.subject_photo_map = pd.read_csv(self.subject_photo_map_file)
        else:
            self.subject_photo_map = None

        self.transform = transform

    def __len__(self):
        return len(self.measurements)

    def __getitem__(self, idx):
        row = self.measurements.iloc[idx]
        subject_id = str(row["subject_id"]) if "subject_id" in row.index else str(row.name)

        # If we have a subject to photo mapping, use it
        if self.subject_photo_map is not None:
            map_row = self.subject_photo_map[self.subject_photo_map["subject_id"] == subject_id]
            if not map_row.empty:
                photo_id = map_row.iloc[0]["photo_id"]
                image_path = os.path.join(self.mask_dir, f"{photo_id}.png")
            else:
                image_path = os.path.join(self.mask_dir, f"{subject_id}.png")
        else:
            # Try to find a matching mask image
            image_path = os.path.join(self.mask_dir, f"{subject_id}.png")

        # If the exact file doesn't exist, try to find other potential matches
        if not os.path.exists(image_path):
            # List all files in the mask directory
            mask_files = os.listdir(self.mask_dir)
            # Check if any file starts with the subject_id
            matching_files = [f for f in mask_files if f.startswith(f"{subject_id}_") or f.startswith(f"{subject_id}.")]
            if matching_files:
                image_path = os.path.join(self.mask_dir, matching_files[0])
            else:
                print(f"Warning: No matching image found for subject {subject_id} in {self.subset_folder}")
                # Return a blank image as fallback
                blank_img = Image.new('RGB', (224, 224), color='black')
                if self.transform:
                    blank_img = self.transform(blank_img)
                # Use default values for measurements
                return blank_img, torch.tensor(20.0, dtype=torch.float32)  # Default body fat percentage

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image as fallback
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        # Extract necessary fields
        height_cm = float(row.get("height", 170))  # Default height if not available
        weight_kg = float(row.get("weight", 70))  # Default weight if not available
        age = float(row.get("age", 30))  # Default age if not available
        gender = row.get("gender", "male").lower()  # Default gender if not available

        # Compute BMI
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)

        # Encode gender: male=1, female=0
        gender_code = 1 if gender == "male" else 0

        # Estimate body fat percentage using Deurenberg formula
        body_fat_percentage = (1.20 * bmi) + (0.23 * age) - (10.8 * gender_code) - 5.4

        return image, torch.tensor(body_fat_percentage, dtype=torch.float32)


# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet means and stds
                         std=[0.229, 0.224, 0.225])
])

# Base path to dataset
base_dir = "./bodym_dataset"

# Create datasets for each subset
train_dataset = BodyMDataset(base_dir=base_dir, subset_folder="train", transform=transform)
testA_dataset = BodyMDataset(base_dir=base_dir, subset_folder="testA", transform=transform)
testB_dataset = BodyMDataset(base_dir=base_dir, subset_folder="testB", transform=transform)

# Combine training and test datasets for training (or use separately as needed)
full_dataset = ConcatDataset([train_dataset, testA_dataset, testB_dataset])
train_dataloader = DataLoader(full_dataset, batch_size=16, shuffle=True)

# You can also create separate dataloaders for validation if needed
test_dataloader = DataLoader(ConcatDataset([testA_dataset, testB_dataset]), batch_size=16, shuffle=False)

# Initialize the model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 1)  # Modify the final layer for regression

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 10
losses = []
model.train()

print(f"Total samples: {len(full_dataset)}")

try:
    for epoch in range(num_epochs):
        total_loss = 0.0
        for i, (images, targets) in enumerate(train_dataloader):
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)  # Add a dimension for regression output

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (i + 1) % 5 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_dataloader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    # Plot the training loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), losses, marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.close()

    # Save the trained model
    torch.save(model.state_dict(), "bodyfat_regressor.pth")
    print("Model training complete and saved as 'bodyfat_regressor.pth'")

    # Evaluation on test data
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images, targets in test_dataloader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)
            outputs = model(images)
            test_loss += criterion(outputs, targets).item()

    avg_test_loss = test_loss / len(test_dataloader)
    print(f"Test Loss: {avg_test_loss:.4f}")

except Exception as e:
    print(f"Error during training: {e}")