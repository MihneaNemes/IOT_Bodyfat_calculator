import torch
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import sys

from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from torchvision.models import resnet50, ResNet50_Weights

# Define global constants for filenames
BEST_MODEL_FILENAME = "best_bodyfat_regressor_model.pth"
NORMALIZATION_PARAMS_FILENAME = "bodyfat_norm_params.npz"

# Model Definition (matching the training code)
class BodyFatRegressor(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
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

# Function to load the trained model and normalization parameters
def load_body_fat_model(model_path, norm_params_path):
    device = torch.device("cpu")  # Using CPU for compatibility
    print(f"Using device: {device}")
    model = BodyFatRegressor(pretrained=True)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
    print(f"Loading model from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    if not os.path.exists(norm_params_path):
        raise FileNotFoundError(f"Normalization parameters file not found at: {norm_params_path}")
    print(f"Loading normalization parameters from: {norm_params_path}")
    params = np.load(norm_params_path)
    target_mean, target_std = params['mean'], params['std']
    return model, device, target_mean, target_std

# Function to get person segmentation mask and bounding box
def get_person_segmentation_details(image_pil, device):
    seg_model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT).to(device)
    seg_model.eval()
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
    binary_person_mask_orig_size = cv2.resize(binary_person_mask, image_pil.size, interpolation=cv2.INTER_NEAREST)
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

# Function to process a single image
def process_image(image_path, device):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    original_pil_image = Image.open(image_path).convert("RGB")
    binary_mask_orig_size, bbox, contours_for_drawing = get_person_segmentation_details(original_pil_image, device)
    if binary_mask_orig_size is None or bbox is None:
        raise ValueError(f"Could not detect a person in the image: {image_path}")
    original_np_image = np.array(original_pil_image)
    binary_mask_3channel = np.stack([binary_mask_orig_size] * 3, axis=-1)
    masked_np_image = original_np_image * binary_mask_3channel
    masked_pil_image = Image.fromarray(masked_np_image)
    cropped_masked_pil_image = masked_pil_image.crop(bbox)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(cropped_masked_pil_image).unsqueeze(0)
    return input_tensor, original_pil_image, masked_pil_image, cropped_masked_pil_image, contours_for_drawing

# Prediction function
def predict_body_fat(front_image_path, side_image_path, height_cm, weight_kg, model_path, norm_params_path):
    try:
        regressor_model, device, target_mean, target_std = load_body_fat_model(model_path, norm_params_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during model loading: {e}")
        import traceback
        traceback.print_exc()
        return

    try:
        print("Processing front image...")
        front_tensor, front_original, front_masked, front_cropped, front_contours = process_image(front_image_path, device)
        front_tensor = front_tensor.to(device)

        print("Processing side image...")
        side_tensor, side_original, side_masked, side_cropped, side_contours = process_image(side_image_path, device)
        side_tensor = side_tensor.to(device)

        print("Predicting body fat percentage...")
        with torch.no_grad():
            normalized_prediction_front = regressor_model(front_tensor)
            normalized_prediction_side = regressor_model(side_tensor)

        predicted_body_fat_front = (normalized_prediction_front.item() * target_std) + target_mean
        predicted_body_fat_side = (normalized_prediction_side.item() * target_std) + target_mean
        predicted_body_fat = (predicted_body_fat_front + predicted_body_fat_side) / 2

        if predicted_body_fat < 6:
            category = "Essential Fat"
        elif predicted_body_fat < 14:
            category = "Athletic"
        elif predicted_body_fat < 18:
            category = "Fitness"
        elif predicted_body_fat < 25:
            category = "Average"
        else:
            category = "Obese"

        print(f"\nPredicted Body Fat: {predicted_body_fat:.2f}% ({category})")
        print(f"Based on images. Height: {height_cm}cm, Weight: {weight_kg}kg (not used in prediction)")

        # Visualization
        plt.figure(figsize=(12, 10))
        plt.subplot(2, 3, 1)
        plt.imshow(front_original)
        plt.title("Front: Original")
        plt.axis("off")
        plt.subplot(2, 3, 2)
        front_with_perimeter = np.array(front_original).copy()
        if front_contours:
            cv2.drawContours(front_with_perimeter, front_contours, -1, (255, 0, 0), 3)
        plt.imshow(front_with_perimeter)
        plt.title("Front: Segmentation")
        plt.axis("off")
        plt.subplot(2, 3, 3)
        plt.imshow(front_cropped)
        plt.title("Front: Processed")
        plt.axis("off")
        plt.subplot(2, 3, 4)
        plt.imshow(side_original)
        plt.title("Side: Original")
        plt.axis("off")
        plt.subplot(2, 3, 5)
        side_with_perimeter = np.array(side_original).copy()
        if side_contours:
            cv2.drawContours(side_with_perimeter, side_contours, -1, (255, 0, 0), 3)
        plt.imshow(side_with_perimeter)
        plt.title("Side: Segmentation")
        plt.axis("off")
        plt.subplot(2, 3, 6)
        plt.imshow(side_cropped)
        plt.title("Side: Processed")
        plt.axis("off")
        plt.suptitle(f"Body Fat Prediction: {predicted_body_fat:.1f}% ({category})\nBased on images. Height: {height_cm}cm, Weight: {weight_kg}kg (not used)", fontsize=16)
        plt.tight_layout()
        plt.show()

        return predicted_body_fat, category
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# Main entry point
if __name__ == "__main__":
    base_dir = r"C:\Users\mihne\OneDrive\Desktop\sarpili\IOT"
    model_file_path = os.path.join(base_dir, "bodym_dataset", "training_outputs", "best_bodyfat_regressor_model.pth")
    norm_params_file_path = os.path.join(base_dir, "bodym_dataset", "training_outputs", "bodyfat_norm_params.npz")
    front_image_path = os.path.join(base_dir, "pictures", "skinny_front.jpg")
    side_image_path = os.path.join(base_dir, "pictures", "skinny_side.jpg")
    height_cm = 175.0
    weight_kg = 70.0

    print(f"Using model file: {model_file_path}")
    print(f"Using params file: {norm_params_file_path}")
    print(f"-tbFront image: {front_image_path}")
    print(f"Side image: {side_image_path}")

    for path in [model_file_path, norm_params_file_path, front_image_path, side_image_path]:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            sys.exit(1)

    predict_body_fat(front_image_path, side_image_path, height_cm, weight_kg, model_file_path, norm_params_file_path)