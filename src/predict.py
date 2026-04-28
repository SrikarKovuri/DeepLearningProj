from pathlib import Path
import argparse

import torch
from PIL import Image

from dataset import get_transforms
from model import build_model, get_device


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "fracture_classifier_resnet18.pth"

IMAGE_SIZE = 224


def predict_image(image_path: str):
    device = get_device()

    model = build_model(num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    transform = get_transforms(image_size=IMAGE_SIZE, train=False)

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        fracture_probability = probabilities[0, 1].item()
        predicted_class = torch.argmax(probabilities, dim=1).item()

    label = "fracture/abnormal" if predicted_class == 1 else "normal"

    return label, fracture_probability


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to X-ray image")
    args = parser.parse_args()

    label, probability = predict_image(args.image)

    print("\nPrediction")
    print("----------")
    print(f"Image: {args.image}")
    print(f"Predicted class: {label}")
    print(f"Fracture/abnormal probability: {probability:.4f}")


if __name__ == "__main__":
    main()