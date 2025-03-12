
# import gradio as gr
# import tensorflow as tf
# import numpy as np
# from PIL import Image

# # Load trained model
# model = tf.keras.models.load_model("model.h5")

# # Define class labels
# class_labels = {0: "Normal", 1: "Pneumonia"}

# # Preprocessing function (compatible with MobileNetV2)
# def preprocess_image(img):
#     img = img.resize((244, 244))  # Resize to match training size
#     img = np.array(img)  # Convert to numpy array
#     img = tf.keras.applications.mobilenet_v2.preprocess_input(img)  # Apply MobileNetV2 preprocessing
#     img = np.expand_dims(img, axis=0)  # Add batch dimension
#     return img

# # Prediction function
# def predict(img):
#     img = preprocess_image(img)
#     prediction = model.predict(img)[0]  # Get prediction
#     class_index = np.argmax(prediction)  # Get highest probability class
#     confidence = float(np.max(prediction))  # Get confidence score
#     return {class_labels[class_index]: confidence}

# # Example Images (Stored in the main directory)
# examples = [
#     ["normal_1.jpg"],
#     ["normal_2.jpg"],
#     ["pneumonia_1.jpg"],
#     ["pneumonia_2.jpg"],
# ]

# # Create Gradio Interface
# interface = gr.Interface(
#     fn=predict,
#     inputs=gr.Image(type="pil"),
#     outputs=gr.Label(),
#     title="Pneumonia Detection",
#     description="Upload a chest X-ray image or select an example below to classify as Normal or Pneumonia.",
#     examples=examples  # Add example images
# )

# # Launch the app
# if __name__ == "__main__":
#     interface.launch()


import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("model.h5")

# Define class labels
class_labels = {0: "🟢 Normal", 1: "🔴 Pneumonia"}

# Preprocessing function (compatible with MobileNetV2)
def preprocess_image(img):
    img = img.resize((244, 244))  # Resize to match training size
    img = np.array(img)  # Convert to numpy array
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)  # Apply MobileNetV2 preprocessing
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Prediction function
def predict(img):
    img = preprocess_image(img)
    prediction = model.predict(img)[0]  # Get prediction
    class_index = np.argmax(prediction)  # Get highest probability class
    confidence = float(np.max(prediction))  # Get confidence score
    return {class_labels[class_index]: round(confidence * 100, 2)}

# Example Images (Stored in the main directory)
examples = [
    ["normal_1.jpg"],
    ["normal_2.jpg"],
    ["pneumonia_1.jpg"],
    ["pneumonia_2.jpg"],
]

# Define Custom CSS
custom_css = """
body {
    background-color: #1e293b;
    color: #e2e8f0;
    font-family: Arial, sans-serif;
}
h1 {
    text-align: center;
    color: #f8fafc;
    font-size: 2.5rem;
}
p {
    text-align: center;
    font-size: 1.2rem;
    color: #cbd5e1;
}
.gradio-container {
    background-color: #334155;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3);
}
"""

# Create Gradio Interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(),
    title="🩺 Pneumonia Detection AI",
    description="**Upload a chest X-ray image or select an example below to classify as Normal or Pneumonia.**\n\n⚡ **Powered by Deep Learning & MobileNetV2**",
    examples=examples,  # Add example images
    theme="default",
    css=custom_css,
)

# Launch the app
if __name__ == "__main__":
    interface.launch()
