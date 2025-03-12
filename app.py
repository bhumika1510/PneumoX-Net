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

# Load the trained model
model = tf.keras.models.load_model("model.keras")

# Example images (stored in the same directory as app.py)
example_images = [
    "normal_1.jpg",
    "normal_2.jpg",
    "pneumonia_1.jpg",
    "pneumonia_2.jpg"
]

def preprocess_image(image):
    """Preprocess the input image to match model requirements."""
    image = image.resize((224, 224))  # Resize to match model input size
    image = np.array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)  # Apply MobileNetV2 preprocessing
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict(image):
    """Predict pneumonia or normal from an input X-ray image."""
    image = preprocess_image(image)
    prediction = model.predict(image)[0]  # Assuming categorical output
    normal_confidence = float(prediction[0])
    pneumonia_confidence = float(prediction[1])
    return {"Normal": normal_confidence, "Pneumonia": pneumonia_confidence}

# Gradio Interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="🩺 Pneumonia Detection",
    description="Upload a chest X-ray to check for pneumonia. Try the example images below!",
    examples=[[img] for img in example_images],  # Convert to list of lists for Gradio compatibility
    theme="default",
    allow_flagging="never"
)

demo.launch()
