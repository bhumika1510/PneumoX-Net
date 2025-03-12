# import gradio as gr
# import tensorflow as tf
# import numpy as np
# from PIL import Image

# # Load the trained model
# model = tf.keras.models.load_model("model.keras")

# def preprocess_image(image):
#     """Preprocess the input image to match model requirements."""
#     image = image.resize((244, 244))  # Resize to match model input size
#     image = np.array(image)
#     image = tf.keras.applications.mobilenet_v2.preprocess_input(image)  # Apply MobileNetV2 preprocessing
#     image = np.expand_dims(image, axis=0)  # Add batch dimension
#     return image

# def predict(image):
#     """Predict pneumonia or normal from an input X-ray image."""
#     image = preprocess_image(image)
#     prediction = model.predict(image)[0]  # Assuming categorical output
#     class_index = np.argmax(prediction)
#     confidence = float(np.max(prediction))
#     label = "Pneumonia" if class_index == 1 else "Normal"
#     return {label: confidence}

# # Gradio Interface
# demo = gr.Interface(
#     fn=predict,
#     inputs=gr.Image(type="pil"),
#     outputs=gr.Label(),
#     title="Pneumonia Detection",
#     description="Upload a chest X-ray to check for pneumonia.",
# )

# demo.launch()

import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("model.h5")

# Define class labels
class_labels = {0: "Normal", 1: "Pneumonia"}

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
    return {class_labels[class_index]: confidence}

# Example Images (Stored in the main directory)
examples = [
    ["normal_1.jpg"],
    ["normal_2.jpg"],
    ["pneumonia_1.jpg"],
    ["pneumonia_2.jpg"],
]

# Create Gradio Interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(),
    title="Pneumonia Detection",
    description="Upload a chest X-ray image or select an example below to classify as Normal or Pneumonia.",
    examples=examples  # Add example images
)

# Launch the app
if __name__ == "__main__":
    interface.launch()
