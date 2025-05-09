# predict_body_fat.py
import torch
from torchvision import models, transforms
from PIL import Image
import os

# Model setup
from torchvision.models import ResNet50_Weights
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Linear(model.fc.in_features, 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("bodyfat_regressor.pth", map_location=device))
model = model.to(device)
model.eval()

# Prediction function
def predict_body_fat(image_path):
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(input_tensor)
    return prediction.item()

# Example usage
if __name__ == "__main__":
    image_path = r"C:\Users\mihne\OneDrive\Desktop\sarpili\IOT\pictures\fat.jpg"
    if os.path.exists(image_path):
        result = predict_body_fat(image_path)
        print(f"Predicted Body Fat Percentage: {result:.2f}%")
    else:
        print(f"Image not found at {image_path}")
