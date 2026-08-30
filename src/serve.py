# flask app for serving the trained model
# has /predict for predictions and /health for kubernetes probes

import io
import os

import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request
from PIL import Image
from torchvision import transforms

from model import get_model

app = Flask(__name__)

# cifar10 class names in order
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "/app/checkpoints/classifier_v1.pt")

# same normalization values as training, important they match!
preprocess = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    ),
])

model = None


def load_model():
    global model
    print("loading checkpoint from", CHECKPOINT_PATH)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    arch = checkpoint.get("architecture", "resnet18")
    num_classes = checkpoint.get("num_classes", 10)
    m = get_model(architecture=arch, num_classes=num_classes)
    m.load_state_dict(checkpoint["model_state_dict"])
    m.eval()
    model = m
    print("model loaded ok, val_accuracy was", checkpoint.get("val_accuracy"))


@app.route("/health", methods=["GET"])
def health():
    # kubernetes calls this to check if the pod is alive
    if model is None:
        return jsonify({"status": "model not loaded"}), 503
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "model not loaded"}), 503

    if "image" not in request.files:
        return jsonify({"error": "no image file in request, use -F image=@file.png"}), 400

    file = request.files["image"]
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": "could not read image: " + str(e)}), 400

    x = preprocess(img).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]

    # build the response with all class probabilities
    result = {CLASSES[i]: round(probs[i].item(), 4) for i in range(len(CLASSES))}
    top_idx = int(probs.argmax())

    return jsonify({
        "predicted_class": CLASSES[top_idx],
        "confidence": round(probs[top_idx].item(), 4),
        "probabilities": result,
    })


# load model when the app starts
try:
    load_model()
except Exception as e:
    print("WARNING: could not load model on startup:", e)
    print("health check will fail until checkpoint is available")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
