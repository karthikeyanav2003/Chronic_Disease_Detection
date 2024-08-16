import streamlit as st
import streamlit as st1
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import pandas as pd
import gdown
from streamlit_navigation_bar import st_navbar
import os

# Define the model class
class RetinalModel(nn.Module):
    def __init__(self, num_parameters):
        super(RetinalModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_parameters)
    
    def forward(self, x):
        return self.resnet(x)

# Download model function
def download_model():
    url = "https://drive.google.com/uc?export=download&id=1nbJUE_P74egDQLfTb4qIdY6AtyqkTadM"
    output = "/mount/src/chronic_disease_detection/best_model_parameters.pth"
    gdown.download(url, output, quiet=False)

# Load the model
def load_model(model_path, num_parameters):
    model = RetinalModel(num_parameters)
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True), strict=False)
        model.eval()
        return model
    except RuntimeError as e:
        print(f"Error loading model state dict: {e}")
        return None

# Preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Clip predictions
def clip_predictions(predictions, ranges):
    return np.clip(predictions, [v[0] for v in ranges.values()], [v[1] for v in ranges.values()])

# Define healthy ranges
healthy_ranges = {
    'Total Cholesterol': (125, 200),
    'LDL': (0, 100),
    'HDL': (40, 100),
    'Triglycerides': (0, 150),
    'Mean Arterial Blood Pressure': (70, 105),
    'eGFR': (90, 150),
    'Albumin': (3.5, 5.0),
    'Fasting Glucose Level': (70, 99),
    'Normal HbA1c': (0, 5.7),
    'Postprandial Glucose Level': (0, 140),
    'Sodium': (135, 145),
    'Potassium': (3.5, 5.0),
    'Red Blood Cells Count': (4.2, 6.1),
    'White Blood Cells Count': (4500, 11000),
    'Packed Cell Volume': (36.1, 50.3),
    'Magnesium': (1.7, 2.2),
    'Uric Acid': (2.6, 7.2),
    'C-Reactive Protein (CRP)': (0.1, 1.0),
    'Body Mass Index (BMI)': (18.5, 24.9),
    'Vitamin D': (20, 50),
    'Systolic Blood Pressure': (90, 120),
    'Diastolic Blood Pressure': (60, 80)
}

def home_page():
    st.markdown(
        """
        <style>
            
            .home-box {
                background-color: #020ac6;
                padding:  40px; /* Increased top and bottom padding for a larger box */
                margin: 20px;  /* Margin to provide space around the box */
                border-radius: 10px;
                color: white;
                font-family: 'Montserrat', sans-serif;
                width: 130%; /* Full width of the container */
                max-width: 1000px; /* Maximum width for large screens */
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Optional: Add shadow for better visibility */
    
                margin-left: auto;
                margin-right: auto;
            }

            .home-box h1, .home-box h2, .home-box p {
                color: white;
            }
            .home-box h1 {
                color: white;
                
                font-weight: 300; /* Thinner font weight */
                font-size: 5rem; /* Increase the size of the title */
            }
            .home-box h2 {
                font-weight: 400; /* Slightly thicker font weight for subheading */
            }
            .home-box p {
                font-weight: 300; /* Thinner font weight for paragraph */
            }
            .col h2,.col p {
                color: black;
            }
        </style>
        <div class="home-box">
            <h1>Welcome to PrediScans - AI in MedTech</h1>
            <p>Your trusted partner in leveraging AI and advanced technology in the field of healthcare. We specialize in implementing cutting-edge AI solutions to revolutionize medical diagnostics and patient care.</p>
        </div>
        <div class="col">
                <h2>About Us</h2>
                <p>At PrediScans, our mission is to transform healthcare through innovative AI technologies.
                    We are dedicated to improving diagnostic accuracy and patient outcomes with our advanced AI
                    solutions.
                    Our team of experts works tirelessly to bring the latest in AI technology to healthcare
                    professionals
                    and patients worldwide.</p>
            </div>
        """, unsafe_allow_html=True
    )



    
def prediction_page():
    st.title("Chronic Disease Detection")
    st.markdown(
        """
        <style>
        .st-emotion-cache-15p99lu {
            display: inline-flex;
            -webkit-box-align: center;
            align-items: center;
            -webkit-box-pack: center;
            justify-content: center;
            font-weight: 400;
            padding: 0.75rem 2.75rem;
            border-radius: 0.5rem;
            min-height: 2.5rem;
            margin: 0px;
            line-height: 1.6;
            color: white;
            width: auto;
            user-select: none;
            background-color: #020ac6;
            border: 1px solid rgba(0, 0, 0, 0.2);
            
            margin-top: 30px; /* Space above the button */
            
        }
        .st-emotion-cache-15p99lu:hover {
            background-color: white; /* Darker blue for hover */
            border-color: black; /* Darker border on hover */
            color : black;
        }
        [role="button"] {
            cursor: pointer;
            margin-bottom: 10px;
            padding: 75px; /* Increase padding to make the box larger */
            font-size: 18px; /* Increase font size for better readability */
            border-radius: 5px; /* Add rounded corners for better aesthetics */
            background-color: #f0f1f1; /* Green background for the button */
            color: black; /* White text color */
            display: block; /* Ensure the button spans the full width */
            text-align: center; /* Center the text inside the button */
            margin-top: 10px; /* Space above the button */
        }
        .st-emotion-cache-1ew6ni3 p {
            word-break: break-word;
            margin-bottom: 0px;
            font-size: 24px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

   
    

    # Download and load the model
    
    model_path = "/mount/src/chronic_disease_detection/best_model_parameters.pth"
    if not os.path.exists(model_path):
        download_model()
    model = load_model(model_path, num_parameters=len(healthy_ranges))
    if model is None:
        st.error("Failed to load model.")
        return
    
    # User input
    st.header("Patient Information")
    name = st.text_input("Enter Patient Name")
    age = st.number_input("Enter Patient Age", min_value=0)
    gender = st.selectbox("Select Gender", ["Male", "Female"])
    
    # Upload images side by side
    st.header("Upload Retinal Images")
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_left_image = st.file_uploader("Upload Left Retinal Image", type=["jpg", "jpeg", "png"], key="left_image")
    
    with col2:
        uploaded_right_image = st.file_uploader("Upload Right Retinal Image", type=["jpg", "jpeg", "png"], key="right_image")

    # Custom HTML for buttons
    if st.button("Predict", key="predict_button"):
        if uploaded_left_image and uploaded_right_image and name and age and gender:
            # Process images
            left_image = Image.open(uploaded_left_image).convert("RGB")
            right_image = Image.open(uploaded_right_image).convert("RGB")
            
            left_image_tensor = preprocess_image(left_image)
            right_image_tensor = preprocess_image(right_image)
            
            # Predict
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device)
            
            left_image_tensor = left_image_tensor.to(device)
            right_image_tensor = right_image_tensor.to(device)
            
            with torch.no_grad():
                left_prediction = model(left_image_tensor).cpu().numpy().flatten()
                right_prediction = model(right_image_tensor).cpu().numpy().flatten()
            
            # Average predictions
            average_prediction = (left_prediction + right_prediction) / 2
            averaged_prediction = clip_predictions(average_prediction, healthy_ranges)
            
            # Output results
            result_df = pd.DataFrame([averaged_prediction], columns=list(healthy_ranges.keys()))
            result_df.insert(0, "Name", [name])
            result_df.insert(1, "Age", [age])
            result_df.insert(2, "Gender", [gender])
            result_df.to_csv('predicted_parameters.csv', index=False)
            
             # Convert all values to strings for consistency
            formatted_predictions = [f"{float(val):.2f}" for val in averaged_prediction]
            #name_str = str(name)
            #age_str = str(age)
            #gender_str = str(gender)
            
            # Create a DataFrame with parameters as rows
            result_df1 = pd.DataFrame({
                'Parameter': list(healthy_ranges.keys()), #+ ['Name', 'Age', 'Gender'],
                'Value': formatted_predictions #+ [name_str, age_str, gender_str]
            })
            # Output results
            st.subheader("Predicted Parameters:")
            st.dataframe(result_df1, use_container_width=True)
            
            st.download_button(
                label="Download CSV",
                data=result_df.to_csv(index=False).encode('utf-8'),
                file_name='predicted_parameters.csv',
                mime='text/csv',
                key="download_button"
            )
        else:
            st.error("Please upload both images and fill out all fields.")  
def main():
    st.set_page_config(initial_sidebar_state="expanded")  # Sidebar open by default

    # Display the logo at the top of the sidebar
    with st.sidebar:
        st.image("logo.png", use_column_width=True)  # Replace with your logo path

        # Add custom CSS for sidebar text links
        st1.markdown("""
        <style>
        .st-emotion-cache-1kideh2 {
            display: inline-flex;
            -webkit-box-align: center;
            align-items: center;
            -webkit-box-pack: center;
            justify-content: center;
            font-weight: 1900;
            padding: 0.25rem 0.75rem;
            border-radius: 0.5rem;
            min-height: 2.5rem;
            margin: 0px;
            line-height: 1.6;
            color: inherit;
            width: auto;
            user-select: none;
            background-color: #f0f1f1;
            border: 0px solid #f0f1f1;
        }
        .st-emotion-cache-1kideh2:focus:not(:active) {
             border-color: #f0f1f1; */
             color: rgb(2, 10, 198); */
             font-weight: 1900;        
        }
        .st-emotion-cache-lr61t1 p {
            word-break: break-word;
            margin-bottom: 0px;
            font-size: 20px;
        }             
        
                     
        </style>
        """, unsafe_allow_html=True)

        # Sidebar navigation buttons
        if st1.button('Home', key='home', help="Go to Home Page"):
            st.session_state.page = 'Home'
        if st1.button('Prediction', key='prediction', help="Go to Prediction Page"):
            st.session_state.page = 'Prediction'

    # Initialize page state if not present
    if 'page' not in st.session_state:
        st.session_state.page = 'Home'

    # Render the selected page
    if st.session_state.page == "Home":
        home_page()
    elif st.session_state.page == "Prediction":
        prediction_page()


if __name__ == "__main__":
    main()
