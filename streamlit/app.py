import streamlit as st
import numpy as np
import tensorflow.lite as tflite
import cv2
from PIL import Image
from fpdf import FPDF
import io

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

# Function to generate and download PDF report
def generate_pdf(image, result_text):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", style='B', size=16)
    pdf.cell(200, 10, "Car Damage Detection Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"Detection Result:\n{result_text}")
    pdf.ln(10)

    # Convert image to RGB if necessary (JPEG does not support RGBA)
    if image.mode == "RGBA":
        image = image.convert("RGB")

    # Convert the image to a temporary buffer
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="JPEG")
    img_buffer.seek(0)

    # Save the image temporarily
    temp_img_path = "temp_uploaded_image.jpg"
    with open(temp_img_path, "wb") as f:
        f.write(img_buffer.read())

    # Add the image to the PDF
    pdf.image(temp_img_path, x=10, y=pdf.get_y(), w=100)

    # Save the PDF to a buffer
    pdf_buffer = io.BytesIO()
    pdf_output = pdf.output(dest="S").encode("latin1")  # Returns PDF as bytes
    pdf_buffer.write(pdf_output)
    pdf_buffer.seek(0)

    return pdf_buffer


# Streamlit UI
def main():

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
        st.image(image, caption="📷 Uploaded Image")

        # Preprocess image
        processed_img = preprocess_image(image)

        if processed_img is not None:
            # Predict
            damage_probability = predict_damage(processed_img)

            # Display results
            if damage_probability is not None:
                st.subheader("🔍 Prediction:")
                if damage_probability < 0.5:
                    st.error(f"🚨 **Car is damaged!** (Confidence: {1-damage_probability:.2%})")
                    result_text = f"Car is damaged! (Confidence: {1-damage_probability:.2%})"
                else:
                    st.success(f"✅ **Car is not damaged.** (Confidence: {damage_probability:.2%})")
                    result_text = f"Car is not damaged. (Confidence: {damage_probability:.2%})"

                # Generate and provide a download link for the PDF report
                pdf_buffer = generate_pdf(image, result_text)
                st.download_button(
                    label="📄 Download Report as PDF",
                    data=pdf_buffer,
                    file_name="Car_Damage_Report.pdf",
                    mime="application/pdf"
                )
    else:
        st.warning("⚠️ Please upload or take a picture before proceeding.")

if __name__ == "__main__":
    main()
