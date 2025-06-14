import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os
from pathlib import Path
import json
from utils.detection_utils import load_model, detect_objects, draw_detections

st.set_page_config(page_title="Object Detection", page_icon="🎯", layout="wide")

st.title("🎯 Object Detection")
st.markdown("Test your trained models on new images or live camera feed")

# Initialize session state
if 'detection_model' not in st.session_state:
    st.session_state.detection_model = None
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = None

# Detection modes
tab1, tab2, tab3, tab4 = st.tabs(["Model Selection", "Image Detection", "Camera Detection", "Batch Processing"])

with tab1:
    st.header("Model Selection")
    
    # Available models
    models_path = Path('models')
    model_files = []
    
    if models_path.exists():
        model_files = [f for f in os.listdir(models_path) if f.endswith('.pt')]
    
    # Pre-trained models option
    pretrained_models = [
        "yolov8n.pt",
        "yolov8s.pt", 
        "yolov8m.pt",
        "yolov8l.pt",
        "yolov8x.pt"
    ]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        model_source = st.radio("Model Source", ["Trained Models", "Pre-trained Models"])
        
        if model_source == "Trained Models":
            if model_files:
                selected_model = st.selectbox("Select Trained Model", model_files)
                model_path = str(models_path / selected_model)
            else:
                st.warning("No trained models found. Train a model first or use pre-trained models.")
                st.stop()
        else:
            selected_model = st.selectbox("Select Pre-trained Model", pretrained_models)
            model_path = selected_model
        
        # Model configuration
        st.subheader("Detection Settings")
        
        col1_1, col1_2 = st.columns(2)
        
        with col1_1:
            confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.5)
            iou_threshold = st.slider("IoU Threshold", 0.1, 0.9, 0.45)
        
        with col1_2:
            max_detections = st.number_input("Max Detections", 1, 100, 20)
            image_size = st.selectbox("Input Image Size", [416, 512, 640, 832, 1024], index=2)
        
        # Load model
        if st.button("🔧 Load Model"):
            with st.spinner("Loading model..."):
                try:
                    model = load_model(model_path)
                    st.session_state.detection_model = model
                    st.session_state.model_path = model_path
                    st.session_state.detection_config = {
                        'confidence': confidence_threshold,
                        'iou': iou_threshold,
                        'max_det': max_detections,
                        'imgsz': image_size
                    }
                    st.success(f"Model loaded successfully: {selected_model}")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
                    if "Ultralytics not installed" in str(e):
                        st.info("To use real object detection models, install ultralytics: `pip install ultralytics`")
                        st.info("For now, you can explore the interface and upload datasets.")
    
    with col2:
        st.subheader("Model Info")
        
        if st.session_state.detection_model is not None:
            st.success("✅ Model Loaded")
            
            # Model details
            model_info = {
                "Model": st.session_state.model_path.split('/')[-1],
                "Status": "Ready",
                "Input Size": f"{st.session_state.detection_config['imgsz']}x{st.session_state.detection_config['imgsz']}"
            }
            
            for key, value in model_info.items():
                st.metric(key, value)
            
            # Detection parameters
            st.markdown("**Detection Parameters:**")
            config = st.session_state.detection_config
            st.write(f"- Confidence: {config['confidence']}")
            st.write(f"- IoU Threshold: {config['iou']}")
            st.write(f"- Max Detections: {config['max_det']}")
        else:
            st.info("No model loaded")
            st.write("Select and load a model to start detection")

with tab2:
    st.header("Image Detection")
    
    if st.session_state.detection_model is None:
        st.warning("Please load a model first in the Model Selection tab.")
        st.stop()
    
    # Image upload
    uploaded_image = st.file_uploader(
        "Upload Image for Detection",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image to detect objects"
    )
    
    if uploaded_image is not None:
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            original_image = Image.open(uploaded_image)
            st.image(original_image, use_container_width=True)
        
        # Run detection
        if st.button("🎯 Detect Objects", type="primary"):
            with st.spinner("Detecting objects..."):
                try:
                    # Convert PIL to numpy array
                    img_array = np.array(original_image)
                    
                    # Run detection
                    results = detect_objects(
                        st.session_state.detection_model,
                        img_array,
                        st.session_state.detection_config
                    )
                    
                    # Draw detections
                    result_image = draw_detections(img_array, results)
                    
                    with col2:
                        st.subheader("Detection Results")
                        st.image(result_image, use_container_width=True)
                    
                    # Detection statistics
                    st.subheader("Detection Statistics")
                    
                    if results and len(results) > 0:
                        # Extract detection info
                        detections = []
                        for result in results:
                            if hasattr(result, 'boxes') and result.boxes is not None:
                                for box in result.boxes:
                                    detections.append({
                                        'class': int(box.cls.item()) if hasattr(box, 'cls') else 0,
                                        'confidence': float(box.conf.item()) if hasattr(box, 'conf') else 0.0,
                                        'bbox': box.xyxy.tolist()[0] if hasattr(box, 'xyxy') else [0, 0, 0, 0]
                                    })
                        
                        if detections:
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Total Detections", len(detections))
                            
                            with col2:
                                avg_confidence = sum(d['confidence'] for d in detections) / len(detections)
                                st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                            
                            with col3:
                                unique_classes = len(set(d['class'] for d in detections))
                                st.metric("Unique Classes", unique_classes)
                            
                            # Detection details table
                            st.subheader("Detection Details")
                            detection_data = []
                            for i, det in enumerate(detections):
                                detection_data.append({
                                    'ID': i + 1,
                                    'Class': f"Class {det['class']}",
                                    'Confidence': f"{det['confidence']:.3f}",
                                    'X1': int(det['bbox'][0]),
                                    'Y1': int(det['bbox'][1]),
                                    'X2': int(det['bbox'][2]),
                                    'Y2': int(det['bbox'][3])
                                })
                            
                            st.dataframe(detection_data, use_container_width=True)
                        else:
                            st.info("No objects detected in the image.")
                    else:
                        st.info("No objects detected in the image.")
                        
                except Exception as e:
                    st.error(f"Error during detection: {str(e)}")

with tab3:
    st.header("Camera Detection")
    
    if st.session_state.detection_model is None:
        st.warning("Please load a model first in the Model Selection tab.")
        st.stop()
    
    st.info("🚧 Camera detection functionality would be implemented here.")
    st.markdown("""
    **Camera Detection Features:**
    - Real-time object detection from webcam
    - Live video stream with bounding boxes
    - Real-time statistics and metrics
    - Recording capabilities
    - Performance monitoring
    
    **Implementation Notes:**
    - Would use OpenCV for camera access
    - Streamlit's camera_input for simple capture
    - WebRTC for real-time streaming
    - Frame rate optimization for smooth performance
    """)
    
    # Simulated camera interface
    camera_image = st.camera_input("Take a picture for detection")
    
    if camera_image is not None:
        st.subheader("Camera Detection Result")
        
        # Process camera image
        image = Image.open(camera_image)
        st.image(image, caption="Captured Image", use_container_width=True)
        
        if st.button("🎯 Detect in Camera Image"):
            with st.spinner("Processing camera image..."):
                try:
                    img_array = np.array(image)
                    results = detect_objects(
                        st.session_state.detection_model,
                        img_array,
                        st.session_state.detection_config
                    )
                    result_image = draw_detections(img_array, results)
                    st.image(result_image, caption="Detection Results", use_container_width=True)
                except Exception as e:
                    st.error(f"Error processing camera image: {str(e)}")

with tab4:
    st.header("Batch Processing")
    
    if st.session_state.detection_model is None:
        st.warning("Please load a model first in the Model Selection tab.")
        st.stop()
    
    # Batch upload
    uploaded_files = st.file_uploader(
        "Upload Multiple Images",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Upload multiple images for batch processing"
    )
    
    if uploaded_files:
        st.write(f"Selected {len(uploaded_files)} images for processing")
        
        # Batch processing settings
        col1, col2 = st.columns(2)
        
        with col1:
            save_results = st.checkbox("Save Detection Results", value=True)
            show_progress = st.checkbox("Show Processing Progress", value=True)
        
        with col2:
            output_format = st.selectbox("Output Format", ["Images with Boxes", "JSON Results", "Both"])
            max_display = st.number_input("Max Images to Display", 1, 20, 5)
        
        if st.button("🔄 Process Batch", type="primary"):
            if show_progress:
                progress_bar = st.progress(0)
            results_container = st.container()
            
            batch_results = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                if show_progress:
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                try:
                    # Load and process image
                    image = Image.open(uploaded_file)
                    img_array = np.array(image)
                    
                    # Run detection
                    results = detect_objects(
                        st.session_state.detection_model,
                        img_array,
                        st.session_state.detection_config
                    )
                    
                    # Store results
                    file_results = {
                        'filename': uploaded_file.name,
                        'detections': [],
                        'image_size': image.size
                    }
                    
                    if results and len(results) > 0:
                        for result in results:
                            if hasattr(result, 'boxes') and result.boxes is not None:
                                for box in result.boxes:
                                    file_results['detections'].append({
                                        'class': int(box.cls.item()) if hasattr(box, 'cls') else 0,
                                        'confidence': float(box.conf.item()) if hasattr(box, 'conf') else 0.0,
                                        'bbox': box.xyxy.tolist()[0] if hasattr(box, 'xyxy') else [0, 0, 0, 0]
                                    })
                    
                    batch_results.append(file_results)
                    
                    # Display first few results
                    if i < max_display:
                        with results_container:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.image(image, caption=uploaded_file.name, use_container_width=True)
                            
                            with col2:
                                if file_results['detections']:
                                    result_image = draw_detections(img_array, results)
                                    st.image(result_image, caption=f"Detections: {len(file_results['detections'])}", use_container_width=True)
                                else:
                                    st.info("No objects detected")
                
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            
            # Batch summary
            st.subheader("Batch Processing Summary")
            
            total_detections = sum(len(r['detections']) for r in batch_results)
            successful_images = len([r for r in batch_results if r['detections']])
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Images", len(batch_results))
            with col2:
                st.metric("Images with Detections", successful_images)
            with col3:
                st.metric("Total Detections", total_detections)
            with col4:
                avg_detections = total_detections / len(batch_results) if batch_results else 0
                st.metric("Avg Detections/Image", f"{avg_detections:.1f}")
            
            # Download results
            if save_results:
                results_json = json.dumps(batch_results, indent=2)
                st.download_button(
                    label="📥 Download Results (JSON)",
                    data=results_json,
                    file_name="batch_detection_results.json",
                    mime="application/json"
                )

# Detection history
st.markdown("---")
st.subheader("Detection History")

# This would store and display previous detection results
st.info("Detection history feature would be implemented to track and review past detection results.")
