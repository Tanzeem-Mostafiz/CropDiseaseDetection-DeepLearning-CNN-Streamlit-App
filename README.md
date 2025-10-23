# 🌱 AGROGUIDE – Crop Disease Detection

**AGROGUIDE** is a web-based application developed as part of a **university course group project**.  
It uses **deep learning (TensorFlow/Keras)** and **computer vision** to detect crop diseases in real time from images or webcam feed. The application provides **disease identification**, **causal explanations**, and **practical cure suggestions** for multiple crops, aiming to support farmers and agricultural researchers.

---

## Project Overview

### Features
- Detect diseases in crops such as **Corn, Pepper, Potato, Tomato, and Cotton**.  
- Upload images or capture live images via webcam.  
- Provide **disease prediction**, **disease description**, and **actionable cure points**.  
- Real-time predictions with a clean, user-friendly **Streamlit interface**.  
- Pre-trained **deep learning model** trained on labeled crop images.

### Crops & Diseases Covered
- **Corn**: Common Rust, Gray Leaf Spot, Northern Leaf Blight  
- **Pepper (Bell)**: Bacterial Spot  
- **Potato**: Early Blight, Late Blight  
- **Tomato**: Late Blight, Yellow Leaf Curl Virus  
- **Cotton**: Bacterial Blight, Curl Virus, Fusarium Wilt  

---

## Technologies Used

| Layer | Technologies |
|-------|---------------|
| **Frontend** | Streamlit, HTML, CSS |
| **Backend** | Python, TensorFlow/Keras, Pillow, NumPy |
| **Computer Vision** | Image preprocessing, real-time webcam feed |
| **Machine Learning** | Convolutional Neural Networks for classification |

---

## Folder Structure
```bash
AGROGUIDE/
│
├── images1/                # Dataset images for training and testing
├── interface/              # UI-related images
├── class_indices.json      # Mapping of class indices to disease labels
├── indexx.py               # Main application script
└── README.md               # This file
```

---

## Installation & Usage
1. Clone the repository
```bash
git clone https://github.com/Tanzeem-Mostafiz/CropDiseaseDetection-DeepLearning-CNN-Streamlit-App
```
2. **Create a virtual environment**
```bash
python -m venv .venv
```
3. **Activate the virtual environment**
 - Windows
```bash
.venv\Scripts\activate
```
 - macOS/Linux
```bash
source .venv/bin/activate
```
4. Run the Streamlit app
```bash
streamlit run indexx.py
```
5. Use the app
   - Upload a crop image or open the camera to capture an image.
   - Click Predict to view disease, explanation, and recommended cure points.

---

## Developer
1. **Tanzeem Mostafiz** | Undergraduate Student, Department of Information and Communication Technology, Bangladesh University of Professionals (BUP)
2. **Aysha Siddika Prity** | Undergraduate Student, Department of Information and Communication Technology, Bangladesh University of Professionals (BUP)

---

## License

This project was developed solely for **academic and educational purposes** as part of a university course.  
You are free to view, reference, and modify the code for **learning or research use**, provided that proper credit is given to the original authors.

© 2024 **Tanzeem Mostafiz** & **Aysha Siddika Prity**. All rights reserved.

---

## Acknowledgment

This project was completed under the guidance and supervision of faculty members from the **Department of Information and Communication Technology**, **Bangladesh University of Professionals (BUP)**.  
We express our sincere gratitude for their valuable support and insights throughout the development process.

