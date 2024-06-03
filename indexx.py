import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Set up the working directory and model path
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/result_model.h5"

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Load the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# Cure points and reasons for each disease
cure_points = {
    "Corn___Common_Rust": [
        "Apply fungicides if the infection is severe and conditions are favorable for disease spread.",
        "Plant rust-resistant corn hybrids.",
        "Rotate crops and manage crop residues to reduce inoculum sources."
    ],
    "Corn___Gray_Leaf_Spot": [
        "Use resistant corn hybrids to reduce disease incidence.",
        "Apply foliar fungicides at the early onset of disease symptoms.",
        "Rotate crops and manage residue to reduce disease pressure."
    ],
    "Corn___Healthy": [
        "Maintain proper fertilization and irrigation practices."
    ],
    "Corn___Northern_Leaf_Blight": [
        "Plant resistant corn varieties to prevent infection.",
        "Apply fungicides at the onset of symptoms if necessary.",
        "Rotate crops and manage crop debris to reduce disease incidence."
    ],
    "Pepper__bell___Bacterial_spot": [
        "Use copper-based fungicides to manage bacterial spot.",
        "Rotate crops to prevent the buildup of the pathogen in the soil.",
        "Ensure proper spacing and pruning for good air circulation."
    ],
    "Pepper__bell___healthy": [
        "Keep plants well-watered but avoid waterlogging."
    ],
    "cotton__bacterial_blight": [
        "Use disease-free seeds and resistant varieties.",
        "Remove and destroy infected plant debris.",
        "Apply appropriate bactericides as per local guidelines."
    ],
    "cotton__curl_virus": [
        "Use virus-free planting material.",
        "Control whitefly populations to prevent virus transmission.",
        "Implement crop rotation and field sanitation."
    ],
    "cotton__fussarium_wilt": [
        "Plant resistant cotton varieties.",
        "Ensure proper field drainage and avoid waterlogging.",
        "Apply soil fumigants if recommended by agricultural experts."
    ],
    "cotton__healthy": [
        "Regularly inspect for pests and diseases."
    ]
}

reasons = {
    "Corn___Common_Rust": "Common Rust in corn is caused by the fungus Puccinia sorghi. It appears as small, circular to elongate, cinnamon-brown pustules on leaves.",
    "Corn___Gray_Leaf_Spot": "Gray Leaf Spot is a fungal disease caused by Cercospora zeae-maydis. Symptoms include rectangular lesions that are tan to gray in color.",
    "Corn___Healthy": "The corn plant is healthy, showing no visible symptoms of diseases or nutrient deficiencies.",
    "Corn___Northern_Leaf_Blight": "Northern Leaf Blight, caused by Exserohilum turcicum, manifests as cigar-shaped, gray-green lesions on leaves.",
    "Pepper__bell___Bacterial_spot": "Bacterial Spot in peppers is caused by Xanthomonas campestris. It leads to water-soaked lesions that turn brown and necrotic.",
    "Pepper__bell___healthy": "The pepper plant is healthy and free from bacterial spot and other diseases.",
    "cotton__bacterial_blight": "Bacterial Blight in cotton is caused by Xanthomonas axonopodis, resulting in angular, water-soaked lesions on leaves.",
    "cotton__curl_virus": "Cotton Curl Virus is transmitted by whiteflies, causing leaf curl, stunted growth, and yellowing of cotton plants.",
    "cotton__fussarium_wilt": "Fusarium Wilt in cotton is caused by Fusarium oxysporum, leading to yellowing, wilting, and browning of leaves.",
    "cotton__healthy": "The cotton plant is healthy and free from any visible disease symptoms."
}

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image):
    # Resize the image
    img = image.resize((224, 224))
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image, class_indices):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    def transform(self, frame):
        self.frame = frame.to_image()
        return frame

    def get_frame(self):
        return self.frame

# Streamlit App
st.markdown(
    """
    <style>
    .stApp {
        background: url('https://c4.wallpaperflare.com/wallpaper/962/839/690/plains-landscape-wallpaper-preview.jpg') no-repeat center center fixed;
        background-size: cover;
    }
    .title {
        color: white;
        font-size: 2em;
        font-weight: bold;
    }
    .file-upload-text {
        color: white;
        font-size: 1.2em;
    }
    .button {
        color: white;
        font-size: 1.2em;
        font-weight: bold;
    }
    .prediction-text {
        color: yellow;
        font-size: 1.2em;
        font-weight: bold;
    }
    .reason-text {
        color: white;
        font-style: italic;
    }
    .cure-points {
        color: white;
        font-size: 1em;
        font-weight: bold;

    }
    .prediction-box {
        background-color: rgba(0, 0, 0, 0.6);
        padding: 10px;
        border-radius: 10px;
    }
    .navbar {
        background-color: rgba(76, 175, 80, 0.5);
        overflow: hidden;
        border-radius: 15px;
        margin-bottom: 20px;
    }
    .navbar a {
        float: left;
        display: block;
        color: white;
        text-align: center;
        padding: 14px 20px;
        padding-left: 220px;
        text-decoration: none;
        font-weight: bold;
        font-size: 3em;
    }
    .navbar-center {
        float: none;
        display: block;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Navbar
st.markdown(
    """
    <div class="navbar">
        <a class="navbar-center">AGROGUIDE</a>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="title">Crop Disease Detection</h1>', unsafe_allow_html=True)

# File uploader for image upload
st.markdown('<p class="file-upload-text">Upload an Image</p>', unsafe_allow_html=True)
uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png"])

# Initialize webcam capture
if "start_camera" not in st.session_state:
    st.session_state["start_camera"] = False
if "captured_image" not in st.session_state:
    st.session_state["captured_image"] = None

if st.button('Open Camera'):
    st.session_state["start_camera"] = True

if st.session_state["start_camera"]:
    ctx = webrtc_streamer(key="example", video_transformer_factory=VideoTransformer, rtc_configuration={"video_constraints": {"width": {"exact": 320}, "height": {"exact": 240}}})

    if ctx.video_transformer:
        if st.button('Capture Image'):
            st.session_state["captured_image"] = ctx.video_transformer.get_frame()
            st.session_state["start_camera"] = False

if st.session_state["captured_image"]:
    st.image(st.session_state["captured_image"], caption='Captured Image', use_column_width=True)
    if st.button('Predict Captured Image'):
        prediction = predict_image_class(model, st.session_state["captured_image"], class_indices)
        reason = reasons[prediction]
        st.markdown(
            f"""
            <div class="prediction-box">
                <p class="prediction-text">Prediction: {str(prediction)}</p>
                <p class="reason-text">: {reason}</p>
                <p class="cure-points">Cure Points:</p>
                {''.join([f'<p class="cure-points">- {point}</p>' for point in cure_points[prediction]])}
            </div>
            """,
            unsafe_allow_html=True
        )

# Display uploaded image and predict
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img, caption='Uploaded Image')

    with col2:
        if st.button('Predict Uploaded Image'):
            prediction = predict_image_class(model, image, class_indices)
            reason = reasons[prediction]
            st.markdown(
                f"""
                <div class="prediction-box">
                    <p class="prediction-text">Prediction: {str(prediction)}</p>
                    <p class="reason-text">: {reason}</p>
                    <p class="cure-points">Cure Points:</p>
                    {''.join([f'<p class="cure-points">- {point}</p>' for point in cure_points[prediction]])}
                </div>
                """,
                unsafe_allow_html=True
            )
