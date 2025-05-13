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

# Define global constants for filenames
BEST_MODEL_FILENAME = "best_bodyfat_regressor_model.pth"
NORMALIZATION_PARAMS_FILENAME = "bodyfat_norm_params.npz"


# Function to get person segmentation mask and bounding box from an image
def get_person_segmentation_details(image_pil, device):
    # Load DeepLabV3 model pre-trained on COCO for segmentation
    seg_model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT).to(device)
    seg_model.eval()

    # Transformation for the segmentation model input
    seg_transform = transforms.Compose([
        transforms.Resize((520, 520)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = seg_transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = seg_model(input_tensor)['out'][0]
    mask_pred = output.argmax(0).cpu().numpy()
    person_class_id = 15  # COCO class ID for 'person'
    binary_person_mask = (mask_pred == person_class_id).astype(np.uint8)

    # Resize binary_person_mask to the original image_pil dimensions
    binary_person_mask_orig_size = cv2.resize(binary_person_mask, image_pil.size, interpolation=cv2.INTER_NEAREST)

    # Find contours to get bounding box on the original sized mask
    contours, _ = cv2.findContours(binary_person_mask_orig_size, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, None  # No person detected

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Add some padding to the bounding box (optional)
    padding = int(0.05 * max(w, h))  # 5% padding
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image_pil.width, x + w + padding)
    y2 = min(image_pil.height, y + h + padding)

    if w == 0 or h == 0 or x2 <= x1 or y2 <= y1:  # Check for invalid bbox
        return None, None, None

    bbox = (x1, y1, x2, y2)
    # Return the binary mask (original size), bounding box, and contours for drawing
    return binary_person_mask_orig_size, bbox, contours


# Function to preprocess images with segmentation
def preprocess_with_segmentation(image_pil, transform, device):
    mask, bbox, _ = get_person_segmentation_details(image_pil, device)
    if bbox is not None:
        # Crop to person bounding box
        cropped_img = image_pil.crop(bbox)
        return transform(cropped_img)
    else:
        # Fallback to original image
        return transform(image_pil)


# Add missing Dataset class
class BodyFatDataset(Dataset):
    def __init__(self, base_dir, split, transform=None, target_mean=None, target_std=None, use_segmentation=True):
        self.base_dir = base_dir
        self.split = split
        self.transform = transform
        self.target_mean = target_mean
        self.target_std = target_std
        self.use_segmentation = use_segmentation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Verify split directory exists
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            raise ValueError(f"Dataset split directory {split} not found in {base_dir}")

        # Load and merge data
        measurements_path = os.path.join(base_dir, split, "measurements.csv")
        hwg_metadata_path = os.path.join(base_dir, split, "hwg_metadata.csv")

        if not os.path.exists(measurements_path):
            raise FileNotFoundError(f"Measurements file not found: {measurements_path}")
        if not os.path.exists(hwg_metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {hwg_metadata_path}")

        measurements_df = pd.read_csv(measurements_path)
        metadata_df = pd.read_csv(hwg_metadata_path)

        self.metadata = pd.merge(measurements_df, metadata_df, on="subject_id")

        # Analyze gender distribution
        if 'gender' in self.metadata.columns:
            gender_dist = self.metadata['gender'].value_counts()
            print(f"Gender distribution in {split} set: {gender_dist}")

        # Calculate body fat percentage using modified U.S. Navy formula
        print(f"Warning: Using wrist/thigh measurements as proxies for neck circumference in body fat calculation")
        self.targets = self.calculate_body_fat()

        # Normalize if parameters provided
        if target_mean is not None and target_std is not None:
            self.targets = (self.targets - target_mean) / target_std

    def calculate_body_fat(self):
        # Convert measurements to inches
        waist_cm = self.metadata['waist']
        height_cm = self.metadata['height_cm']
        weight_kg = self.metadata['weight_kg']

        # US Navy formula constants (using waist-to-height ratio as neck proxy)
        body_fat_percentage = []
        for idx, row in self.metadata.iterrows():
            waist = waist_cm[idx] / 2.54  # Convert cm to inches
            height = height_cm[idx] / 2.54

            if row['gender'] == 'male':
                # Male formula: 86.010×log10(waist - neck) - 70.041×log10(height) + 36.76
                # Using wrist as neck proxy since neck measurement unavailable
                bfp = (86.010 * np.log10(waist - (row['wrist'] / 2.54)) -
                       70.041 * np.log10(height) + 30.76)
            else:
                # Female formula: 163.205×log10(waist + hip - neck) - 97.684×log10(height) - 78.387
                # Using thigh as neck proxy
                hip = row['hip'] / 2.54
                bfp = (163.205 * np.log10(waist + hip - (row['thigh'] / 2.54)) -
                       97.684 * np.log10(height) - 68.387)

            body_fat_percentage.append(bfp)

        return np.clip(np.array(body_fat_percentage), 5, 50)  # Keep within realistic range

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Get image path from subject-photo mapping
        subject_id = self.metadata.iloc[idx]['subject_id']
        map_path = os.path.join(self.base_dir, self.split, "subject_to_photo_map.csv")

        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Subject to photo mapping file not found: {map_path}")

        subject_map = pd.read_csv(map_path)

        # Fix: Check which column exists in the subject_map
        if 'photo_filename' in subject_map.columns:
            img_name = subject_map[subject_map['subject_id'] == subject_id]['photo_filename'].values[0]
        elif 'photo_id' in subject_map.columns:
            photo_id = subject_map[subject_map['subject_id'] == subject_id]['photo_id'].values[0]
            img_name = f"{photo_id}.png"
        else:
            # Fallback to using subject_id directly
            img_name = f"{subject_id}.png"

        # Fix: Check if 'mask_left' directory exists, otherwise use 'mask'
        mask_dir = "mask_left" if os.path.exists(os.path.join(self.base_dir, self.split, "mask_left")) else "mask"
        img_path = os.path.join(self.base_dir, self.split, mask_dir, img_name)

        # Handle file not found errors gracefully
        try:
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
            else:
                print(f"Warning: Image not found at {img_path}. Using black placeholder.")
                image = Image.new("RGB", (224, 224), color="black")
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            image = Image.new("RGB", (224, 224), color="black")

        target = self.targets[idx]

        # Apply segmentation if enabled
        if self.use_segmentation and self.transform:
            try:
                transformed_image = preprocess_with_segmentation(image, self.transform, self.device)
            except Exception as e:
                print(f"Segmentation error for {img_path}: {str(e)}. Using standard transform.")
                transformed_image = self.transform(image) if self.transform else image
        elif self.transform:
            transformed_image = self.transform(image)
        else:
            transformed_image = image

        return transformed_image, torch.tensor(target, dtype=torch.float32)


# Model Definition
class BodyFatRegressor(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Load base ResNet50 model
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = resnet50(weights=weights)

        self.features = torch.nn.Sequential(
            *list(base_model.children())[:-2],
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.regressor(x)


# Define transformed subset class OUTSIDE of main function to make it picklable
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


# Training Configuration
def main():
    # --- Configuration ---
    base_dir = r"C:\Users\mihne\OneDrive\Desktop\sarpili\IOT\bodym_dataset"  # Use raw string for paths
    output_dir = os.path.join(base_dir, "training_outputs")
    os.makedirs(output_dir, exist_ok=True)
    model_save_path = os.path.join(output_dir, BEST_MODEL_FILENAME)
    norm_params_save_path = os.path.join(output_dir, NORMALIZATION_PARAMS_FILENAME)

    # Add configuration flag for segmentation
    use_segmentation = True  # Set to False to disable segmentation

    # --- Print configuration summary ---
    print("\n=== Configuration ===")
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Use segmentation: {use_segmentation}")
    print("=====================\n")

    # --- Data transforms ---
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

    # --- Initialize datasets ---
    print("Loading initial training data to calculate normalization parameters...")
    try:
        initial_train_dataset_for_norm = BodyFatDataset(
            base_dir, "train", transform=None, use_segmentation=False
        )
        target_mean = initial_train_dataset_for_norm.targets.mean()
        target_std = initial_train_dataset_for_norm.targets.std()
        if target_std == 0:
            print("Error: Calculated target_std is 0. Check your target data. Exiting.")
            return
        np.savez(norm_params_save_path, mean=target_mean, std=target_std)
        print(
            f"Saved target normalization parameters (mean={target_mean:.2f}, std={target_std:.2f}) to: {norm_params_save_path}")

        full_train_val_dataset = BodyFatDataset(
            base_dir, "train", transform=None,
            target_mean=target_mean, target_std=target_std,
            use_segmentation=use_segmentation
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"Error initializing datasets: {str(e)}")
        return

    # --- Data Splitting ---
    train_indices, val_indices = None, None
    if "subject_id" in full_train_val_dataset.metadata.columns and \
            full_train_val_dataset.metadata["subject_id"].nunique() > 1:
        print("Attempting GroupShuffleSplit for train/validation split...")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        groups = full_train_val_dataset.metadata["subject_id"]
        try:
            train_indices, val_indices = next(gss.split(X=np.arange(len(full_train_val_dataset)), groups=groups))
            print(f"GroupShuffleSplit successful: Train size: {len(train_indices)}, Val size: {len(val_indices)}")
        except ValueError as e:
            print(
                f"Warning: GroupShuffleSplit failed ({e}). Falling back to random_split. Subject distribution might be skewed.")
            train_indices, val_indices = None, None

    if train_indices is None:
        print(
            "Using random_split for train/validation split. Potential for subject data leakage if subjects have multiple images.")
        total_size = len(full_train_val_dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
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
        print(f"Error loading test datasets: {str(e)}")
        print("Continuing with training only...")
        test_dataset = None

    # --- Create Dataloaders ---
    batch_size = 32
    # Set num_workers=0 to avoid multiprocessing issues on Windows
    num_workers = 0  # This fixes the pickling error
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers) if test_dataset else None

    # --- Initialize model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = BodyFatRegressor(pretrained=True).to(device)

    # --- Training setup ---
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)
    # Fix deprecated verbose parameter by removing it
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)
    criterion = nn.HuberLoss()

    # --- Training loop ---
    num_epochs = 25
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stopping_patience = 10

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        progress_bar_train = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", leave=False)

        for images, targets in progress_bar_train:
            images = images.to(device)
            targets = targets.view(-1, 1).to(device)

            optimizer.zero_grad()

            outputs = model(images)  # Standard forward pass
            loss = criterion(outputs, targets)

            # Standard backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar_train.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0
        progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]", leave=False)
        with torch.no_grad():
            for images, targets in progress_bar_val:
                images = images.to(device)
                targets = targets.view(-1, 1).to(device)
                outputs = model(images)  # Standard forward pass
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                progress_bar_val.set_postfix(loss=f"{loss.item():.4f}")

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"\nEpoch {epoch + 1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss  : {avg_val_loss:.4f}")
        print(
            f"  LR        : {optimizer.param_groups[0]['lr']:.2e}")  # Use param_groups instead of deprecated get_last_lr()

        if avg_val_loss < best_val_loss:
            print(
                f"  Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving model to {model_save_path}")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            epochs_no_improve = 0
        else:
            print(f"  Validation loss did not improve from {best_val_loss:.4f}.")
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered after {early_stopping_patience} epochs without improvement.")
            break
        print("-" * 30)

    # --- Final evaluation on Test Set ---
    if test_loader:
        print("\nLoading best model for final evaluation on test set...")
        try:
            model.load_state_dict(torch.load(model_save_path, map_location=device))
        except FileNotFoundError:
            print(f"ERROR: Best model not found at {model_save_path}. Cannot perform final evaluation.")
            return

        model.eval()
        test_loss = 0
        predictions_denorm = []
        actuals_denorm = []

        def denormalize_target(value, mean, std):
            return (value * std) + mean

        with torch.no_grad():
            for images, targets_norm in tqdm(test_loader, desc="Testing"):
                images = images.to(device)
                targets_norm = targets_norm.view(-1, 1).to(device)
                outputs_norm = model(images)  # Standard forward pass
                loss = criterion(outputs_norm, targets_norm)
                test_loss += loss.item()

                batch_predictions_denorm = denormalize_target(outputs_norm.cpu().numpy(), target_mean, target_std)
                batch_actuals_denorm = denormalize_target(targets_norm.cpu().numpy(), target_mean, target_std)
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