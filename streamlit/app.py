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
    try:
        # Convert PIL image to NumPy array
        image = np.array(image, dtype=np.uint8)

        # Ensure it's in RGB format
        if len(image.shape) == 2:  # If grayscale, convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[-1] == 4:  # If RGBA, convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Ensure the image is a valid NumPy array
        if not isinstance(image, np.ndarray):
            raise ValueError("Error: Image is not a valid NumPy array.")

        # Resize image to match model input size
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

        # Normalize if the model expects float32
        if input_dtype == np.float32:
            image = image.astype(np.float32) / 255.0

        # Expand dimensions to match (1, height, width, 3)
        image = np.expand_dims(image, axis=0)

        return image

    except Exception as e:
        st.error(f"⚠️ Image preprocessing failed: {e}")
        return None


# Function to make a prediction
def predict_damage(image):
    try:
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
        damage_probability = float(prediction.item())  # Ensure correct value extraction

        return damage_probability

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
        return None


# Streamlit UI
st.title("🚗 Car Damage Detector")

# Option to upload or take a picture
option = st.radio("Choose an option:", ["Upload Image", "Take a Photo"])

image = None  # Initialize image variable

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"Error loading uploaded image: {e}")

elif option == "Take a Photo":
    captured_image = st.camera_input("Take a photo")
    if captured_image is not None:
        try:
            image = Image.open(captured_image)
        except Exception as e:
            st.error(f"Error capturing image: {e}")

# Ensure an image was uploaded or captured before proceeding
if image is not None:
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    # Preprocess image
    processed_img = preprocess_image(image)

    if processed_img is not None:
        # Predict
        damage_probability = predict_damage(processed_img)

        # Display results
        if damage_probability is not None:
            st.subheader("🔍 Prediction:")
            if damage_probability < 0.5:
                st.error(f"🚨 **Car is damaged!** (Confidence: {damage_probability:.2%})")
            else:
                st.success(f"✅ **Car is not damaged.** (Confidence: {1 - damage_probability:.2%})")
else:
    st.warning("⚠️ Please upload or take a picture before proceeding.")
