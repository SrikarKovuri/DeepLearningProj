import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Builds a ResNet18 image classifier.

    Output classes:
        0 = normal / no abnormality
        1 = abnormal / possible fracture
    """

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def get_device() -> torch.device:
    """
    Uses Apple Silicon GPU if available, otherwise CUDA if available,
    otherwise CPU.
    """

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")