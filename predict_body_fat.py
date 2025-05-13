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

# Define global constants for filenames (matching the training script)
BEST_MODEL_FILENAME = "best_bodyfat_regressor_model.pth"
NORMALIZATION_PARAMS_FILENAME = "bodyfat_norm_params.npz"


# Model Definition (MUST BE IDENTICAL TO THE ONE USED IN TRAINING)
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


# Function to load the trained BodyFatRegressor model and normalization parameters
def load_body_fat_model(model_path, norm_params_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model
    model = BodyFatRegressor(pretrained=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
    print(f"Loading model from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Load normalization parameters (mean, std) for the target variable
    if not os.path.exists(norm_params_path):
        raise FileNotFoundError(f"Normalization parameters file not found at: {norm_params_path}")
    print(f"Loading normalization parameters from: {norm_params_path}")
    params = np.load(norm_params_path)
    target_mean, target_std = params['mean'], params['std']

    return model, device, target_mean, target_std


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


# Main prediction and display function
def predict_body_fat_from_image(image_path, model_path, norm_params_path):
    try:
        # Load the trained body fat regression model and normalization params
        regressor_model, device, target_mean, target_std = load_body_fat_model(model_path, norm_params_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during model loading: {e}")
        import traceback
        traceback.print_exc()
        return

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    try:
        original_pil_image = Image.open(image_path).convert("RGB")

        # --- 1. Get Person Segmentation Mask and Bounding Box ---
        print("Segmenting person from image...")
        binary_mask_orig_size, bbox, contours_for_drawing = get_person_segmentation_details(original_pil_image, device)

        if binary_mask_orig_size is None or bbox is None:
            print("Could not detect a person in the image. Cannot proceed with prediction.")
            # Display original image if no person found
            plt.imshow(original_pil_image)
            plt.title("Original Image (No person detected)")
            plt.axis("off")
            plt.show()
            return

        # --- 2. Apply Mask to Original Image ---
        # Convert original PIL image to NumPy array
        original_np_image = np.array(original_pil_image)
        # Ensure mask is 3-channel for multiplication if original_np_image is 3-channel
        binary_mask_3channel = np.stack([binary_mask_orig_size] * 3, axis=-1)
        # Apply mask: background becomes black
        masked_np_image = original_np_image * binary_mask_3channel
        # Convert NumPy masked image back to PIL Image
        masked_pil_image = Image.fromarray(masked_np_image)

        # --- 3. Crop the Masked Person using Bounding Box ---
        cropped_masked_pil_image = masked_pil_image.crop(bbox)

        # --- 4. Prepare Cropped Masked Image for Regressor Model ---
        # This transform should match the validation/test transform from training
        regressor_input_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = regressor_input_transform(cropped_masked_pil_image).unsqueeze(0).to(device)

        # --- 5. Predict Body Fat ---
        print("Predicting body fat...")
        with torch.no_grad():
            normalized_prediction = regressor_model(input_tensor)

        # Denormalize the prediction using the loaded target mean and std
        predicted_body_fat = (normalized_prediction.item() * target_std) + target_mean

        # --- Categorize Body Fat (example categories) ---
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

        # --- Visualization ---
        plt.figure(figsize=(18, 6))

        # Original Image
        plt.subplot(1, 4, 1)
        plt.imshow(original_pil_image)
        plt.title("Original Image")
        plt.axis("off")

        # Original Image with Segmentation Perimeter
        plt.subplot(1, 4, 2)
        image_with_perimeter = original_np_image.copy()  # Draw on a copy
        if contours_for_drawing:
            cv2.drawContours(image_with_perimeter, contours_for_drawing, -1, (255, 0, 0), 3)  # Red perimeter
        plt.imshow(image_with_perimeter)
        plt.title("Segmentation Perimeter")
        plt.axis("off")

        # Masked Image (before crop)
        plt.subplot(1, 4, 3)
        plt.imshow(masked_pil_image)
        plt.title("Mask Applied (Background Black)")
        plt.axis("off")

        # Cropped Masked Image (Input to Regressor)
        plt.subplot(1, 4, 4)
        plt.imshow(cropped_masked_pil_image)
        plt.title(f"Cropped Input to Model\nPred: {predicted_body_fat:.1f}% ({category})")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An error occurred during prediction/visualization: {e}")
        import traceback
        traceback.print_exc()


# Function to allow prediction from command line with an image path
def predict_from_cli():
    import argparse
    parser = argparse.ArgumentParser(description="Predict body fat percentage from an image")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--model_dir", help="Directory containing the model files",
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_outputs"))
    args = parser.parse_args()

    model_path = os.path.join(args.model_dir, BEST_MODEL_FILENAME)
    norm_params_path = os.path.join(args.model_dir, NORMALIZATION_PARAMS_FILENAME)

    predict_body_fat_from_image(args.image_path, model_path, norm_params_path)


# --- Main execution block ---
if __name__ == "__main__":
    # --- Configuration ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dataset_dir = os.path.dirname(script_dir)
    if os.path.basename(base_dataset_dir) != "IOT":
        base_dataset_dir = r"C:\Users\mihne\OneDrive\Desktop\sarpili\IOT"

    bodym_dataset_dir = os.path.join(base_dataset_dir, "bodym_dataset")
    training_outputs_dir = os.path.join(bodym_dataset_dir, "training_outputs")
    pictures_dir = os.path.join(bodym_dataset_dir, "pictures")

    os.makedirs(training_outputs_dir, exist_ok=True)
    os.makedirs(pictures_dir, exist_ok=True)

    model_file_path = os.path.join(training_outputs_dir, BEST_MODEL_FILENAME)
    norm_params_file_path = os.path.join(training_outputs_dir, NORMALIZATION_PARAMS_FILENAME)

    # Hardcoded image path
    test_image_path = r"C:\Users\mihne\OneDrive\Desktop\sarpili\IOT\pictures\skinny.jpg"

    print(f"Predicting body fat for image: {test_image_path}")
    print(f"Using model: {model_file_path}")
    print(f"Using normalization params: {norm_params_file_path}")

    if not os.path.exists(model_file_path):
        print(f"ERROR: Model file not found: {model_file_path}")
        sys.exit(1)
    elif not os.path.exists(norm_params_file_path):
        print(f"ERROR: Normalization parameters file not found: {norm_params_file_path}")
        sys.exit(1)

    predict_body_fat_from_image(test_image_path, model_file_path, norm_params_file_path)
