import streamlit as st
import json
import os

def save_feedback(name, email, comment):
    feedback_data = {"name": name, "email": email, "comment": comment}
    
    # Ensure the feedback directory exists
    feedback_dir = "feedback_storage"
    os.makedirs(feedback_dir, exist_ok=True)
    
    # Save feedback as a JSON file
    feedback_file = os.path.join(feedback_dir, "feedback_data.json")
    
    # Load existing feedback if available
    if os.path.exists(feedback_file):
        with open(feedback_file, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    
    # Append new feedback and save
    data.append(feedback_data)
    with open(feedback_file, "w") as f:
        json.dump(data, f, indent=4)
    
    return True

def main():
    st.title("📩 Contact Us / Feedback")
    
    st.write("We value your feedback! Please fill out the form below to reach us.")
    
    # Form Inputs
    name = st.text_input("Your Name")
    email = st.text_input("Your Email")
    comment = st.text_area("Your Message")
    
    # Ensure all fields are filled
    if st.button("Submit Feedback"):
        if name.strip() == "" or email.strip() == "" or comment.strip() == "":
            st.error("⚠️ All fields are required!")
        else:
            if save_feedback(name, email, comment):
                st.success("✅ Feedback submitted successfully! Thank you for reaching out.")

                # Clear fields after submission
                st.session_state.name = ""
                st.session_state.email = ""
                st.session_state.comment = ""
                st.rerun()

if __name__ == "__main__":
    main()

