import streamlit as st
import numpy as np
import tensorflow.lite as tflite
import tensorflow as tf
import cv2
from PIL import Image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Load the TFLite damage detection model
@st.cache_resource  # Cache the interpreter for efficiency
def load_damage_model():
    interpreter = tflite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

damage_interpreter = load_damage_model()

# Load EfficientNetB0 for car verification (pretrained on ImageNet)
@st.cache_resource
def load_car_model():
    return EfficientNetB0(weights="imagenet")

car_model = load_car_model()

# Get input details for damage model preprocessing
input_details = damage_interpreter.get_input_details()
output_details = damage_interpreter.get_output_details()

# Extract input shape & dtype
input_shape = input_details[0]['shape']  # (1, height, width, 3)
input_dtype = input_details[0]['dtype']
height, width = input_shape[1], input_shape[2]

# Function to check if the image contains a car
def is_car_image(image):
    img = image.resize((224, 224))  # Resize for EfficientNetB0
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # Normalize

    # Predict class
    preds = car_model.predict(img_array)
    decoded_preds = decode_predictions(preds, top=3)[0]  # Get top 3 predictions

    # Debugging: Display top predictions
    st.write("🔍 *Car Model Predictions:*", [(label, f"{prob*100:.2f}%") for (_, label, prob) in decoded_preds])

    # Check if any of the top predictions indicate a car
    car_labels = ["sports_car", "SUV", "convertible", "jeep", "pickup", "limousine", "cab", "car_wheel"]
    return any(label in car_labels for _, label, _ in decoded_preds)

# Function to preprocess image for damage detection
def preprocess_image(image):
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

# Function to make a prediction for damage detection
def predict_damage(image):
    input_tensor_index = input_details[0]['index']
    damage_interpreter.set_tensor(input_tensor_index, image)

    # Run inference
    damage_interpreter.invoke()

    # Get output tensor index
    output_tensor_index = output_details[0]['index']
    prediction = damage_interpreter.get_tensor(output_tensor_index)

    # Extract probability correctly
    damage_probability = prediction.item()

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

# If an image is uploaded, proceed
if 'image' in locals():
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Step 1: Check if the image is a car
    if is_car_image(image):
        # Step 2: Preprocess image for damage detection
        processed_img = preprocess_image(image)

        # Step 3: Predict damage
        damage_probability = predict_damage(processed_img)

        # Display results
        st.subheader("🔍 Prediction:")
        
        if damage_probability < 0.5:
            st.error(f"🚨 *Car is damaged!* (Confidence: {damage_probability:.2%})")
        else:
            st.success(f"✅ *Car is not damaged.* (Confidence: {1 - damage_probability:.2%})")
    else:
        st.warning("⚠️ Please upload an image that contains a *car*.")
