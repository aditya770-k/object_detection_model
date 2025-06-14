import streamlit as st
import os
from pathlib import Path
from utils.database import init_database, get_database_stats

# Set page configuration
st.set_page_config(
    page_title="Object Detection Trainer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create necessary directories
def create_directories():
    """Create necessary directories for the application"""
    directories = ['models', 'datasets', 'temp']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

# Initialize the application
create_directories()

# Initialize database
try:
    init_database()
    db_connected = True
except Exception as e:
    st.error(f"Database connection failed: {str(e)}")
    db_connected = False

# Main page content
st.title("🎯 Object Detection Trainer")
st.markdown("### Train custom object detection models and deploy with Roboflow")

# Introduction
st.markdown("""
Welcome to the Object Detection Trainer! This application provides a complete workflow for:

- **Dataset Management**: Upload and organize your custom image datasets
- **Model Training**: Train YOLOv8 models on your custom data
- **Object Detection**: Test your trained models on new images
- **Roboflow Integration**: Deploy your models to Roboflow for production use

Use the sidebar to navigate between different sections of the application.
""")

# Quick stats
col1, col2, col3, col4 = st.columns(4)

if db_connected:
    try:
        stats = get_database_stats()
        with col1:
            st.metric("Datasets", stats.get('total_datasets', 0))
        with col2:
            st.metric("Trained Models", stats.get('total_models', 0))
        with col3:
            st.metric("Training Jobs", stats.get('total_training_jobs', 0))
        with col4:
            st.metric("Detections", stats.get('total_detections', 0))
    except Exception as e:
        with col1:
            st.metric("Datasets", len([d for d in os.listdir('datasets') if os.path.isdir(os.path.join('datasets', d))]))
        with col2:
            st.metric("Trained Models", len([f for f in os.listdir('models') if f.endswith('.pt')]))
        with col3:
            st.metric("Supported Formats", "JPG, PNG")
        with col4:
            st.metric("Framework", "YOLOv8")
else:
    with col1:
        st.metric("Datasets", len([d for d in os.listdir('datasets') if os.path.isdir(os.path.join('datasets', d))]))
    with col2:
        st.metric("Trained Models", len([f for f in os.listdir('models') if f.endswith('.pt')]))
    with col3:
        st.metric("Supported Formats", "JPG, PNG")
    with col4:
        st.metric("Framework", "YOLOv8")

# Getting started guide
st.markdown("---")
st.markdown("## 🚀 Getting Started")

st.markdown("""
1. **Prepare Your Dataset**: Go to Dataset Management to upload your images and annotations
2. **Train Your Model**: Use the Model Training page to train a YOLOv8 model
3. **Test Detection**: Try your trained model on new images in the Object Detection page
4. **Deploy**: Use Roboflow Integration to deploy your model for production use

For best results, ensure your dataset has:
- At least 100-300 images per class for simple models
- Diverse lighting conditions and backgrounds
- Properly annotated bounding boxes
- Balanced class distribution
""")

# System requirements
with st.expander("System Requirements & Tips"):
    st.markdown("""
    **Minimum Requirements:**
    - Python 3.8+
    - 4GB RAM (8GB+ recommended for training)
    - GPU support recommended for faster training
    
    **Supported Image Formats:**
    - JPEG (.jpg, .jpeg)
    - PNG (.png)
    
    **Annotation Formats:**
    - YOLO format (.txt files)
    - COCO JSON format
    - Pascal VOC XML format
    
    **Tips for Better Results:**
    - Use high-quality images (min 640x640 resolution)
    - Include diverse scenarios in your dataset
    - Validate annotations before training
    - Start with pre-trained weights when possible
    """)
