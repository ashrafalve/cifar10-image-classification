"""Gradio application for CIFAR-10 image classification."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gradio as gr

from src.predict import Predictor


CHECKPOINT_PATHS = [
    "checkpoints/best_model.pth",
    "checkpoints/resnet18/best_model.pth",
]


def find_checkpoint():
    for path in CHECKPOINT_PATHS:
        if os.path.isfile(path):
            return path
    return None


def classify_image(image):
    if image is None:
        return "Please upload an image.", {}

    checkpoint_path = find_checkpoint()
    if checkpoint_path is None:
        return "No checkpoint found. Please train a model first.", {}

    predictor = Predictor(checkpoint_path=checkpoint_path)
    result = predictor.predict(image)

    top = result["top_prediction"]
    top_text = f"**{top['class'].capitalize()}** ({top['confidence']:.2%})"

    label_dict = {p["class"].capitalize(): p["confidence"] for p in result["top_3_predictions"]}

    return top_text, label_dict


with gr.Blocks(title="CIFAR-10 Image Classifier") as demo:
    gr.Markdown("# CIFAR-10 Image Classifier")
    gr.Markdown("Upload an image to classify it into one of 10 CIFAR-10 categories.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            predict_btn = gr.Button("Classify", variant="primary")

        with gr.Column():
            top_prediction = gr.Markdown(label="Top Prediction")
            confidence_labels = gr.Label(label="Top 3 Confidence Scores", num_top_classes=3)

    predict_btn.click(
        fn=classify_image,
        inputs=image_input,
        outputs=[top_prediction, confidence_labels],
    )


if __name__ == "__main__":
    checkpoint_path = find_checkpoint()
    if checkpoint_path:
        print(f"Using checkpoint: {checkpoint_path}")
    else:
        print("Warning: No checkpoint found. Please train a model first.")

    demo.launch(theme=gr.themes.Soft())
