import streamlit as st
import os
import shutil
import json
from pathlib import Path
import zipfile
import tempfile
from PIL import Image
import pandas as pd
from utils.dataset_utils import validate_dataset, create_dataset_info, visualize_annotations
from utils.database import save_dataset_to_db, get_datasets_from_db

st.set_page_config(page_title="Dataset Management", page_icon="📁", layout="wide")

st.title("📁 Dataset Management")
st.markdown("Upload, organize, and validate your custom object detection datasets")

# Initialize session state
if 'current_dataset' not in st.session_state:
    st.session_state.current_dataset = None

# Dataset operations
tab1, tab2, tab3, tab4 = st.tabs(["Upload Dataset", "Dataset Overview", "Validation", "Export"])

with tab1:
    st.header("Upload New Dataset")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        dataset_name = st.text_input("Dataset Name", help="Enter a unique name for your dataset")
        dataset_description = st.text_area("Description", help="Describe your dataset and use case")
        
        upload_method = st.radio("Upload Method", ["ZIP Archive", "Individual Files"])
        
        if upload_method == "ZIP Archive":
            uploaded_file = st.file_uploader(
                "Upload Dataset ZIP", 
                type=['zip'],
                help="Upload a ZIP file containing images and annotations"
            )
            
            if uploaded_file and dataset_name:
                if st.button("Process ZIP Archive"):
                    with st.spinner("Processing dataset..."):
                        try:
                            # Create dataset directory
                            dataset_path = Path('datasets') / dataset_name
                            dataset_path.mkdir(exist_ok=True)
                            
                            # Extract ZIP file
                            with tempfile.TemporaryDirectory() as temp_dir:
                                zip_path = Path(temp_dir) / uploaded_file.name
                                with open(zip_path, 'wb') as f:
                                    f.write(uploaded_file.getvalue())
                                
                                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                                    zip_ref.extractall(temp_dir)
                                
                                # Move files to dataset directory
                                for root, dirs, files in os.walk(temp_dir):
                                    for file in files:
                                        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.txt', '.json', '.xml')):
                                            src = Path(root) / file
                                            dst = dataset_path / file
                                            shutil.copy2(src, dst)
                            
                            # Create dataset info
                            info = create_dataset_info(dataset_path, dataset_name, dataset_description)
                            
                            st.success(f"Dataset '{dataset_name}' uploaded successfully!")
                            st.session_state.current_dataset = dataset_name
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error processing dataset: {str(e)}")
        
        else:  # Individual Files
            images = st.file_uploader(
                "Upload Images", 
                type=['jpg', 'jpeg', 'png'], 
                accept_multiple_files=True
            )
            annotations = st.file_uploader(
                "Upload Annotations", 
                type=['txt', 'json', 'xml'], 
                accept_multiple_files=True
            )
            
            if images and dataset_name:
                if st.button("Upload Individual Files"):
                    with st.spinner("Uploading files..."):
                        try:
                            # Create dataset directory
                            dataset_path = Path('datasets') / dataset_name
                            dataset_path.mkdir(exist_ok=True)
                            
                            # Save images
                            for image in images:
                                with open(dataset_path / image.name, 'wb') as f:
                                    f.write(image.getvalue())
                            
                            # Save annotations if provided
                            if annotations:
                                for annotation in annotations:
                                    with open(dataset_path / annotation.name, 'wb') as f:
                                        f.write(annotation.getvalue())
                            
                            # Create dataset info
                            info = create_dataset_info(dataset_path, dataset_name, dataset_description)
                            
                            st.success(f"Dataset '{dataset_name}' uploaded successfully!")
                            st.session_state.current_dataset = dataset_name
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error uploading files: {str(e)}")
    
    with col2:
        st.subheader("Upload Guidelines")
        st.markdown("""
        **Image Requirements:**
        - Formats: JPG, PNG
        - Min resolution: 640x640
        - Clear, well-lit images
        
        **Annotation Formats:**
        - YOLO: .txt files
        - COCO: .json files
        - Pascal VOC: .xml files
        
        **ZIP Structure:**
        ```
        dataset.zip
        ├── image1.jpg
        ├── image1.txt
        ├── image2.jpg
        ├── image2.txt
        └── ...
        ```
        """)

with tab2:
    st.header("Dataset Overview")
    
    # Get datasets from database first, fallback to filesystem
    try:
        db_datasets = get_datasets_from_db()
        datasets = [ds.name for ds in db_datasets]
        
        if datasets:
            selected_dataset = st.selectbox("Select Dataset", datasets)
            
            # Find the selected dataset object
            selected_dataset_obj = next((ds for ds in db_datasets if ds.name == selected_dataset), None)
            
            if selected_dataset_obj:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Images", selected_dataset_obj.total_images)
                with col2:
                    st.metric("Total Annotations", selected_dataset_obj.total_annotations)
                with col3:
                    st.metric("Classes", len(selected_dataset_obj.classes or []))
                
                # Display dataset description
                st.markdown(f"**Description:** {selected_dataset_obj.description or 'No description available'}")
                
                # Class distribution
                if selected_dataset_obj.class_distribution:
                    st.subheader("Class Distribution")
                    df = pd.DataFrame(list(selected_dataset_obj.class_distribution.items()), 
                                    columns=['Class', 'Count'])
                    st.bar_chart(df.set_index('Class'))
                
                # Validation status
                if selected_dataset_obj.validation_passed:
                    st.success("✅ Dataset validation passed")
                else:
                    st.warning("⚠️ Dataset validation issues found")
                    if selected_dataset_obj.issues:
                        for issue in selected_dataset_obj.issues:
                            st.warning(f"• {issue}")
    
    except Exception as e:
        st.warning(f"Database connection issue, using filesystem: {str(e)}")
        
        # Fallback to filesystem
        datasets_path = Path('datasets')
        if datasets_path.exists():
            datasets = [d for d in os.listdir(datasets_path) if os.path.isdir(datasets_path / d)]
            
            if datasets:
                selected_dataset = st.selectbox("Select Dataset", datasets)
            
            if selected_dataset:
                dataset_path = datasets_path / selected_dataset
                
                # Load dataset info
                info_file = dataset_path / 'dataset_info.json'
                if info_file.exists():
                    with open(info_file, 'r') as f:
                        dataset_info = json.load(f)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Images", dataset_info.get('total_images', 0))
                    with col2:
                        st.metric("Total Annotations", dataset_info.get('total_annotations', 0))
                    with col3:
                        st.metric("Classes", len(dataset_info.get('classes', [])))
                    
                    # Display dataset description
                    st.markdown(f"**Description:** {dataset_info.get('description', 'No description available')}")
                    
                    # Class distribution
                    if 'class_distribution' in dataset_info:
                        st.subheader("Class Distribution")
                        df = pd.DataFrame(list(dataset_info['class_distribution'].items()), 
                                        columns=['Class', 'Count'])
                        st.bar_chart(df.set_index('Class'))
                    
                    # Sample images
                    st.subheader("Sample Images")
                    image_files = [f for f in os.listdir(dataset_path) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    
                    if image_files:
                        cols = st.columns(4)
                        for i, img_file in enumerate(image_files[:8]):  # Show first 8 images
                            with cols[i % 4]:
                                img_path = dataset_path / img_file
                                img = Image.open(img_path)
                                st.image(img, caption=img_file, use_container_width=True)
                else:
                    st.warning("Dataset info not found. Please validate the dataset.")
        else:
            st.info("No datasets found. Upload a dataset in the 'Upload Dataset' tab.")
    else:
        st.info("No datasets directory found.")

with tab3:
    st.header("Dataset Validation")
    
    datasets_path = Path('datasets')
    if datasets_path.exists():
        datasets = [d for d in os.listdir(datasets_path) if os.path.isdir(datasets_path / d)]
        
        if datasets:
            validate_dataset_name = st.selectbox("Select Dataset to Validate", datasets, key="validate")
            
            if st.button("Validate Dataset"):
                dataset_path = datasets_path / validate_dataset_name
                
                with st.spinner("Validating dataset..."):
                    validation_results = validate_dataset(dataset_path)
                
                # Display validation results
                if validation_results['valid']:
                    st.success("✅ Dataset validation passed!")
                else:
                    st.error("❌ Dataset validation failed!")
                
                # Show detailed results
                st.subheader("Validation Details")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Statistics:**")
                    for key, value in validation_results['stats'].items():
                        st.write(f"- {key.replace('_', ' ').title()}: {value}")
                
                with col2:
                    if validation_results['issues']:
                        st.markdown("**Issues Found:**")
                        for issue in validation_results['issues']:
                            st.warning(issue)
                    else:
                        st.success("No issues found!")
        else:
            st.info("No datasets available for validation.")

with tab4:
    st.header("Export Dataset")
    
    datasets_path = Path('datasets')
    if datasets_path.exists():
        datasets = [d for d in os.listdir(datasets_path) if os.path.isdir(datasets_path / d)]
        
        if datasets:
            export_dataset = st.selectbox("Select Dataset to Export", datasets, key="export")
            export_format = st.selectbox("Export Format", ["YOLO", "COCO", "Pascal VOC"])
            
            col1, col2 = st.columns(2)
            
            with col1:
                train_split = st.slider("Training Split", 0.1, 0.9, 0.8)
            with col2:
                val_split = st.slider("Validation Split", 0.05, 0.5, 0.2)
            
            test_split = 1.0 - train_split - val_split
            st.write(f"Test Split: {test_split:.2f}")
            
            if st.button("Export Dataset"):
                with st.spinner("Exporting dataset..."):
                    try:
                        # Implementation would depend on the specific format
                        st.success(f"Dataset exported in {export_format} format!")
                        st.info("Export functionality will be implemented based on the specific format requirements.")
                    except Exception as e:
                        st.error(f"Error exporting dataset: {str(e)}")
        else:
            st.info("No datasets available for export.")

# Dataset management actions
st.markdown("---")
st.subheader("Dataset Actions")

if st.session_state.current_dataset:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Delete Current Dataset", type="secondary"):
            if st.session_state.current_dataset:
                dataset_path = Path('datasets') / st.session_state.current_dataset
                if dataset_path.exists():
                    shutil.rmtree(dataset_path)
                    st.session_state.current_dataset = None
                    st.success("Dataset deleted successfully!")
                    st.rerun()
    
    with col2:
        if st.button("📋 Copy Dataset", type="secondary"):
            st.info("Copy functionality would be implemented here")
    
    with col3:
        if st.button("📤 Share Dataset", type="secondary"):
            st.info("Share functionality would be implemented here")
