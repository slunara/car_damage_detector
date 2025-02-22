import streamlit as st


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Landing Page", "App", "Contact Us / Feedback"])
    
    if page == "Landing Page":
        st.title("Car Damage Detection App")
        
        st.header("About the Application")
        st.write(
            "This application utilizes a Deep Learning model to detect if a car is damaged or not based on an image. "
            "The model is based on EfficientNetV2S and has been trained using real data provided by the Latin American insurance company Sura. "
            "Sura, a leading insurance provider, leverages AI technology to enhance its claim assessment processes, reducing manual effort, and expediting claim resolutions. "
            "By automating damage assessment through image recognition, Sura can provide faster payouts, reduce fraudulent claims, and optimize operational costs. "
            "This model supports the insurance industry by improving efficiency in car inspections and claim handling, ultimately benefiting both the company and its customers."
        )
        
        st.header("How to Use")
        st.write(
            "1. Navigate to the 'App' tab.\n"
            "2. Upload or take a picture of the car (recommended to use a mobile device for better convenience).\n"
            "3. The model will analyze the image and provide a result indicating whether the car is damaged or not.\n"
            "4. The model is specifically trained to assess vehicle damage, so it will provide an output regardless of the image content.\n"
            "   However, uploading a non-car image may lead to misleading results since the model is not designed to detect objects other than cars.\n"
            "5. Optionally, you can download a PDF report that includes the image and the model's result."
        )
        
        st.header("Group Members")
        st.write("- Andrea Bordon")
        st.write("- Gabriel Chapman")
        st.write("- Leire Díez")
        st.write("- Simón García")
        st.write("- Susana Luna")
        st.write("- Philippa Von Quadt")
        st.write("- Yoel Winer")
    
    elif page == "App":
        import app  # Lazy import to prevent premature execution
        app.main()
    
    elif page == "Contact Us / Feedback":
        import feedback
        feedback.main()

if __name__ == "__main__":
    main()
