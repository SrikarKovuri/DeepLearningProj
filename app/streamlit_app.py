import sys
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from dataset import get_transforms
from model import build_model, get_device


MODEL_PATH = PROJECT_ROOT / "models" / "fracture_classifier_resnet18.pth"
IMAGE_SIZE = 224


@st.cache_resource
def load_model():
    device = get_device()

    model = build_model(num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    return model, device


def predict_image(image):
    model, device = load_model()

    transform = get_transforms(image_size=IMAGE_SIZE, train=False)

    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    normal_prob = probabilities[0, 0].item()
    fracture_prob = probabilities[0, 1].item()

    return normal_prob, fracture_prob


st.set_page_config(
    page_title="X-ray Fracture Classifier",
    page_icon="🩻",
    layout="centered",
)

st.title("🩻 X-ray Fracture Classifier")

st.write(
    """
    Upload a radiograph image and the model will classify it as either:

    **Not fractured** or **Fractured**.

    This is an educational deep learning project only. It is not suitable for real clinical diagnosis.
    """
)

st.divider()

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Train the model first using: `python src/train.py`"
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Upload an X-ray image",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded radiograph", use_container_width=True)

    normal_prob, fracture_prob = predict_image(image)

    st.subheader("Prediction Result")

    if fracture_prob >= 0.5:
        st.error("Prediction: Fractured")
    else:
        st.success("Prediction: Not fractured")

    st.metric("Not fractured probability", f"{normal_prob:.2%}")
    st.metric("Fractured probability", f"{fracture_prob:.2%}")

    st.progress(fracture_prob)

    st.caption(
        "Model output is based on a binary classifier trained on your downloaded fracture dataset."
    )
else:
    st.info("Upload an X-ray image to get a prediction.")
