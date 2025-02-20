import streamlit as st
import numpy as np
import tensorflow.lite as tflite
import cv2
from PIL import Image

# Load the TFLite model
@st.cache_resource  # Cache the interpreter for efficiency
def load_model():
    interpreter = tflite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

# Get input details for preprocessing
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Extract input shape & dtype
input_shape = input_details[0]['shape']  # (1, height, width, 3)
input_dtype = input_details[0]['dtype']
height, width = input_shape[1], input_shape[2]

# Function to preprocess image
def preprocess_image(image):
    # Convert PIL image to numpy array
    image = np.array(image)

    # Convert to RGB if needed
    if image.shape[-1] == 4:  # Handle RGBA images
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif len(image.shape) == 2:  # Grayscale to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Resize to model input size
    image = cv2.resize(image, (width, height))

    # Normalize if model expects float32
    if input_dtype == np.float32:
        image = image.astype(np.float32) / 255.0

    # Expand dimensions to match (1, height, width, 3)
    image = np.expand_dims(image, axis=0)

    return image

# Function to make a prediction
def predict_damage(image):
    # Get input tensor index
    input_tensor_index = input_details[0]['index']
    interpreter.set_tensor(input_tensor_index, image)

    # Run inference
    interpreter.invoke()

    # Get output tensor index
    output_tensor_index = output_details[0]['index']
    prediction = interpreter.get_tensor(output_tensor_index)

    # Debugging: Show raw model output
    st.write("🔍 **Raw Model Output:**", prediction)

    # Extract probability correctly
    damage_probability = prediction.item()  # Ensure correct value extraction

    return damage_probability

# Streamlit UI
st.title("🚗 Car Damage Detector")

# Option to upload or take a picture
option = st.radio("Choose an option:", ["Upload Image", "Take a Photo"])

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
elif option == "Take a Photo":
    image = st.camera_input("Take a photo")
    if image:
        image = Image.open(image)

# If image exists, process and predict
if 'image' in locals():
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    processed_img = preprocess_image(image)

    # Predict
    damage_probability = predict_damage(processed_img)

    # Display results
    st.subheader("🔍 Prediction:")
    
    if damage_probability > 0.5:
        st.error(f"🚨 **Car is damaged!** (Confidence: {damage_probability:.2%})")
    else:
        st.success(f"✅ **Car is not damaged.** (Confidence: {1 - damage_probability:.2%})")
