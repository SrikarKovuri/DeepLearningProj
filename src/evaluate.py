from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from dataset import XrayDataset, get_transforms
from model import build_model, get_device


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_CSV = PROJECT_ROOT / "data" / "splits" / "test.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "fracture_classifier_resnet18.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 16


def main():
    device = get_device()
    print(f"Using device: {device}")

    test_dataset = XrayDataset(
        csv_path=str(TEST_CSV),
        transform=get_transforms(image_size=IMAGE_SIZE, train=False),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = build_model(num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)

            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print("\nEvaluation Results")
    print("------------------")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nConfusion Matrix")
    print(cm)

    print("\nClassification Report")
    print(classification_report(all_labels, all_preds, target_names=["normal", "fracture/abnormal"]))


if __name__ == "__main__":
    main()