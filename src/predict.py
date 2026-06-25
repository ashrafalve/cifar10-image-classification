"""Prediction module for CIFAR-10 image classification."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from PIL import Image

from src.model import get_model
from src.dataset import get_val_transforms
from src.evaluate import CIFAR10_CLASSES


class Predictor:
    def __init__(self, checkpoint_path, model_name="baseline", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = get_model(model_name=model_name, num_classes=10)
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
        else:
            self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        self.transform = get_val_transforms()
        self.class_names = CIFAR10_CLASSES

    def predict(self, image_input):
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise TypeError("image_input must be a file path (str) or PIL.Image.Image")

        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1)

        top_probs, top_indices = torch.topk(probabilities, k=3, dim=1)

        top_predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            top_predictions.append({
                "class": self.class_names[idx.item()],
                "class_index": idx.item(),
                "confidence": float(prob.item()),
            })

        return {
            "top_prediction": top_predictions[0],
            "top_3_predictions": top_predictions,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict CIFAR-10 class from image")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "resnet18"])
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    predictor = Predictor(checkpoint_path=args.checkpoint, model_name=args.model)
    result = predictor.predict(args.image)

    print(f"\nTop prediction : {result['top_prediction']['class']} ({result['top_prediction']['confidence']:.4f})")
    print("Top 3 predictions:")
    for i, pred in enumerate(result["top_3_predictions"], 1):
        print(f"  {i}. {pred['class']:>12} - confidence: {pred['confidence']:.4f}")
