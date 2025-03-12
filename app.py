
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image


model = tf.keras.models.load_model("model.h5")


class_labels = {0: "🟢 Normal", 1: "🔴 Pneumonia"}

# Preprocessing function (compatible with MobileNetV2)
def preprocess_image(img):
    img = img.resize((244, 244))  
    img = np.array(img) 
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)  
    img = np.expand_dims(img, axis=0) 
    return img

# Prediction function
def predict(img):
    img = preprocess_image(img)
    prediction = model.predict(img)[0] 
    class_index = np.argmax(prediction)  
    confidence = float(np.max(prediction))  
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
    title="🩺 Pneumonia Detection AI",
    description="**Upload a chest X-ray image or select an example below to classify as Normal or Pneumonia.**\n\n⚡ **Powered by Deep Learning & MobileNetV2**",
    examples=examples,  # Add example images
    theme="default"
)

# Launch the app
if __name__ == "__main__":
    interface.launch()
