import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models import ResNet50_Weights

# Hard-coded paths
MODEL_PATH = "bodyfat_regressor.pth"
IMAGE_PATH = r"C:\Users\mihne\OneDrive\Desktop\sarpili\IOT\pictures\testimg1.png"  # Hard-coded image path


# Load the model
def load_model():
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use weights_only=True to address the security warning
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model, device


# Categorize body fat percentage
def categorize_body_fat(percentage):
    """Categorize body fat percentage based on general health guidelines."""
    if percentage < 0:
        return "Invalid measurement"

    # Male categories
    if percentage < 5:
        return "essential"
    elif percentage < 13:
        return "athletic"
    elif percentage < 17:
        return "fitness"
    elif percentage < 25:
        return "average"
    else:
        return "obese"


# Predict and display results
def predict_and_display():
    try:
        # Load model
        model, device = load_model()
        print(f"Model loaded successfully. Using device: {device}")

        # Load and process image
        image = Image.open(IMAGE_PATH).convert("RGB")

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor)

        body_fat_percentage = prediction.item()
        category = categorize_body_fat(body_fat_percentage)

        # Print result
        print(f"Predicted Body Fat Percentage: {body_fat_percentage:.2f}% ({category.capitalize()})")

        # Visualization
        plt.figure(figsize=(10, 6))

        # Display the image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Input Image")
        plt.axis('off')

        # Display the category visualization
        plt.subplot(1, 2, 2)
        categories = ["essential", "athletic", "fitness", "average", "obese"]
        category_values = [1, 2, 3, 4, 5]
        colors = ['lightblue', 'green', 'yellow', 'orange', 'red']

        plt.bar(categories, category_values, color=colors)
        current_index = categories.index(category) if category in categories else -1

        if current_index >= 0:
            plt.scatter(current_index, category_values[current_index],
                        color='blue', s=200, zorder=3, marker='*')

        plt.title(f"Body Fat: {body_fat_percentage:.2f}% ({category.capitalize()})")
        plt.ylim(0, 6)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error: {e}")


# Run the prediction
if __name__ == "__main__":
    predict_and_display()