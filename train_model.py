import torch
import os
import numpy as np
from PIL import Image
import pandas as pd
import cv2
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit

from torch import nn, optim
from torch.utils.data import Dataset, ConcatDataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from torchvision.models import resnet50, ResNet50_Weights
import math

# Define global constants for filenames
BEST_MODEL_FILENAME = "best_bodyfat_regressor_model.pth"
NORMALIZATION_PARAMS_FILENAME = "bodyfat_norm_params.npz"

# Function to get person segmentation mask and bounding box from an image
def get_person_segmentation_details(image_pil):
    # Load DeepLabV3 model for segmentation
    seg_model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
    seg_model.eval()

    # Transformation for the segmentation model input
    seg_transform = transforms.Compose([
        transforms.Resize((520, 520)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = seg_transform(image_pil).unsqueeze(0)

    with torch.no_grad():
        output = seg_model(input_tensor)['out'][0]
    mask_pred = output.argmax(0).numpy()
    person_class_id = 15  # COCO class ID for 'person'
    binary_person_mask = (mask_pred == person_class_id).astype(np.uint8)

    # Resize to original dimensions
    binary_person_mask_orig_size = cv2.resize(binary_person_mask, image_pil.size, interpolation=cv2.INTER_NEAREST)

    # Find contours for bounding box
    contours, _ = cv2.findContours(binary_person_mask_orig_size, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, None

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    padding = int(0.05 * max(w, h))
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image_pil.width, x + w + padding)
    y2 = min(image_pil.height, y + h + padding)

    if w == 0 or h == 0 or x2 <= x1 or y2 <= y1:
        return None, None, None

    bbox = (x1, y1, x2, y2)
    return binary_person_mask_orig_size, bbox, contours

# Function to preprocess images with segmentation
def preprocess_with_segmentation(image_pil, transform):
    mask, bbox, _ = get_person_segmentation_details(image_pil)
    if bbox is not None:
        cropped_img = image_pil.crop(bbox)
        return transform(cropped_img)
    else:
        return transform(image_pil)

class BodyFatDataset(Dataset):
    def __init__(self, base_dir, split, transform=None, target_mean=None, target_std=None, use_segmentation=True):
        self.base_dir = base_dir
        self.split = split
        self.transform = transform
        self.target_mean = target_mean
        self.target_std = target_std
        self.use_segmentation = use_segmentation

        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            raise ValueError(f"Dataset split directory {split} not found in {base_dir}")

        measurements_path = os.path.join(base_dir, split, "measurements.csv")
        hwg_metadata_path = os.path.join(base_dir, split, "hwg_metadata.csv")

        if not os.path.exists(measurements_path):
            raise FileNotFoundError(f"Measurements file not found: {measurements_path}")
        if not os.path.exists(hwg_metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {hwg_metadata_path}")

        measurements_df = pd.read_csv(measurements_path)
        metadata_df = pd.read_csv(hwg_metadata_path)

        self.metadata = pd.merge(measurements_df, metadata_df, on="subject_id")

        if 'gender' in self.metadata.columns:
            gender_dist = self.metadata['gender'].value_counts()
            print(f"Gender distribution in {split} set: {gender_dist}")

        print(f"Warning: Using wrist/thigh measurements as proxies for neck circumference in body fat calculation")
        self.targets = self.calculate_body_fat()

        if target_mean is not None and target_std is not None:
            self.targets = (self.targets - target_mean) / target_std

    def calculate_body_fat(self):
        body_fat_percentage = []
        for idx, row in self.metadata.iterrows():
            height_cm = row['height_cm']
            waist_cm = row['waist']
            hip_cm = row.get('hip', np.nan)
            wrist_cm = row.get('wrist', np.nan)
            gender = row['gender']

            height_in = height_cm / 2.54
            waist_in = waist_cm / 2.54
            hip_in = hip_cm / 2.54 if not np.isnan(hip_cm) else np.nan
            wrist_in = wrist_cm / 2.54 if not np.isnan(wrist_cm) else np.nan

            if gender == 'male':
                estimated_neck_in = 2.14 * wrist_in if not np.isnan(wrist_in) else np.nan
            else:
                estimated_neck_in = 2.17 * wrist_in if not np.isnan(wrist_in) else np.nan

            if np.isnan(estimated_neck_in) or np.isnan(waist_in) or np.isnan(height_in):
                body_fat_percentage.append(np.nan)
                continue

            try:
                if gender == 'male':
                    if estimated_neck_in >= waist_in:
                        body_fat_percentage.append(np.nan)
                        continue
                    bf = 86.010 * math.log10(waist_in - estimated_neck_in) - 70.041 * math.log10(height_in) + 36.76
                else:
                    if np.isnan(hip_in):
                        body_fat_percentage.append(np.nan)
                        continue
                    if waist_in + hip_in - estimated_neck_in <= 0:
                        body_fat_percentage.append(np.nan)
                        continue
                    bf = 163.205 * math.log10(waist_in + hip_in - estimated_neck_in) - 97.684 * math.log10(height_in) + 36.76

                bf = np.clip(bf, 5, 60)
                body_fat_percentage.append(bf)
            except ValueError:
                body_fat_percentage.append(np.nan)

        return np.array(body_fat_percentage)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        subject_id = self.metadata.iloc[idx]['subject_id']
        map_path = os.path.join(self.base_dir, self.split, "subject_to_photo_map.csv")

        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Subject to photo mapping file not found: {map_path}")

        subject_map = pd.read_csv(map_path)

        if 'photo_filename' in subject_map.columns:
            img_name = subject_map[subject_map['subject_id'] == subject_id]['photo_filename'].values[0]
        elif 'photo_id' in subject_map.columns:
            photo_id = subject_map[subject_map['subject_id'] == subject_id]['photo_id'].values[0]
            img_name = f"{photo_id}.png"
        else:
            img_name = f"{subject_id}.png"

        mask_dir = "mask_left" if os.path.exists(os.path.join(self.base_dir, self.split, "mask_left")) else "mask"
        img_path = os.path.join(self.base_dir, self.split, mask_dir, img_name)

        try:
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
            else:
                image = Image.new("RGB", (224, 224), color="black")
        except Exception as e:
            image = Image.new("RGB", (224, 224), color="black")

        target = self.targets[idx]

        if self.use_segmentation and self.transform:
            try:
                transformed_image = preprocess_with_segmentation(image, self.transform)
            except Exception as e:
                transformed_image = self.transform(image) if self.transform else image
        elif self.transform:
            transformed_image = self.transform(image)
        else:
            transformed_image = image

        return transformed_image, torch.tensor(target, dtype=torch.float32)

class BodyFatRegressor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = resnet50(weights=weights)

        self.features = nn.Sequential(
            *list(base_model.children())[:-2],
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.regressor = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.regressor(x)

class TransformedSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def main():
    base_dir = r"bodym_dataset"
    output_dir = os.path.join(base_dir, "training_outputs")
    os.makedirs(output_dir, exist_ok=True)
    model_save_path = os.path.join(output_dir, BEST_MODEL_FILENAME)
    norm_params_save_path = os.path.join(output_dir, NORMALIZATION_PARAMS_FILENAME)
    use_segmentation = True

    print("\n=== Configuration ===")
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Use segmentation: {use_segmentation}")
    print("=====================\n")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Loading initial training data to calculate normalization parameters...")
    try:
        initial_train_dataset_for_norm = BodyFatDataset(
            base_dir, "train", transform=None, use_segmentation=False
        )
        target_mean = initial_train_dataset_for_norm.targets.mean()
        target_std = initial_train_dataset_for_norm.targets.std()
        np.savez(norm_params_save_path, mean=target_mean, std=target_std)

        full_train_val_dataset = BodyFatDataset(
            base_dir, "train", transform=None,
            target_mean=target_mean, target_std=target_std,
            use_segmentation=use_segmentation
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"Error initializing datasets: {str(e)}")
        return

    train_indices, val_indices = None, None
    if "subject_id" in full_train_val_dataset.metadata.columns:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        groups = full_train_val_dataset.metadata["subject_id"]
        try:
            train_indices, val_indices = next(gss.split(X=np.arange(len(full_train_val_dataset)), groups=groups))
        except ValueError:
            total_size = len(full_train_val_dataset)
            train_size = int(0.8 * total_size)
            val_size = total_size - train_size
            train_subset_full, val_subset_full = random_split(full_train_val_dataset, [train_size, val_size],
                                                            generator=torch.Generator().manual_seed(42))
    else:
        total_size = len(full_train_val_dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        train_subset_full, val_subset_full = random_split(full_train_val_dataset, [train_size, val_size],
                                                        generator=torch.Generator().manual_seed(42))

    if train_indices is None:
        train_subset_full, val_subset_full = random_split(full_train_val_dataset, [train_size, val_size],
                                                        generator=torch.Generator().manual_seed(42))
    else:
        train_subset_full = torch.utils.data.Subset(full_train_val_dataset, train_indices)
        val_subset_full = torch.utils.data.Subset(full_train_val_dataset, val_indices)

    train_subset = TransformedSubset(train_subset_full, train_transform)
    val_subset = TransformedSubset(val_subset_full, val_test_transform)

    print("Loading test datasets...")
    try:
        testA_dataset = BodyFatDataset(
            base_dir, "testA", transform=val_test_transform,
            target_mean=target_mean, target_std=target_std,
            use_segmentation=use_segmentation
        )
        testB_dataset = BodyFatDataset(
            base_dir, "testB", transform=val_test_transform,
            target_mean=target_mean, target_std=target_std,
            use_segmentation=use_segmentation
        )
        test_dataset = ConcatDataset([testA_dataset, testB_dataset])
    except (ValueError, FileNotFoundError) as e:
        test_dataset = None

    batch_size = 8  # Reduced for CPU memory
    num_workers = 0
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers) if test_dataset else None

    model = BodyFatRegressor(pretrained=True)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)
    criterion = nn.HuberLoss()

    num_epochs = 50
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stopping_patience = 10

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        progress_bar_train = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", leave=False)

        for images, targets in progress_bar_train:
            targets = targets.view(-1, 1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            progress_bar_train.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]", leave=False)
        with torch.no_grad():
            for images, targets in progress_bar_val:
                targets = targets.view(-1, 1)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                progress_bar_val.set_postfix(loss=f"{loss.item():.4f}")

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"\nEpoch {epoch + 1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss  : {avg_val_loss:.4f}")
        print(f"  LR        : {optimizer.param_groups[0]['lr']:.2e}")

        if avg_val_loss < best_val_loss:
            print(f"  Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving model to {model_save_path}")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered after {early_stopping_patience} epochs without improvement.")
            break
        print("-" * 30)

    if test_loader:
        print("\nLoading best model for final evaluation on test set...")
        try:
            model.load_state_dict(torch.load(model_save_path))
        except FileNotFoundError:
            return

        model.eval()
        test_loss = 0
        predictions_denorm = []
        actuals_denorm = []

        def denormalize_target(value, mean, std):
            return (value * std) + mean

        with torch.no_grad():
            for images, targets_norm in tqdm(test_loader, desc="Testing"):
                targets_norm = targets_norm.view(-1, 1)
                outputs_norm = model(images)
                loss = criterion(outputs_norm, targets_norm)
                test_loss += loss.item()

                batch_predictions_denorm = denormalize_target(outputs_norm.numpy(), target_mean, target_std)
                batch_actuals_denorm = denormalize_target(targets_norm.numpy(), target_mean, target_std)
                predictions_denorm.extend(batch_predictions_denorm.flatten())
                actuals_denorm.extend(batch_actuals_denorm.flatten())

        avg_test_loss = test_loss / len(test_loader)

        from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
        mae = mean_absolute_error(actuals_denorm, predictions_denorm)
        r2 = r2_score(actuals_denorm, predictions_denorm)
        rmse = np.sqrt(mean_squared_error(actuals_denorm, predictions_denorm))

        print("\n--- Final Test Results ---")
        print(f"Test Loss (normalized) : {avg_test_loss:.4f}")
        print(f"MAE (denormalized)     : {mae:.2f}% body fat")
        print(f"RMSE (denormalized)    : {rmse:.2f}% body fat")
        print(f"R² Score (denormalized): {r2:.4f}")
        print("\nSample Predictions vs Actuals (denormalized):")
        for i in range(min(5, len(predictions_denorm))):
            print(f"  Pred: {predictions_denorm[i]:.1f}% | Actual: {actuals_denorm[i]:.1f}%")
    else:
        print("\nNo test dataset available for final evaluation.")

if __name__ == "__main__":
    main()
