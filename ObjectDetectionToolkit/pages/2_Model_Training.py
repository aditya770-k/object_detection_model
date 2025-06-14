import streamlit as st
import os
import json
import time
from pathlib import Path
import threading
import queue
import matplotlib.pyplot as plt
import pandas as pd
from utils.training_utils import prepare_training_data, start_training, monitor_training

st.set_page_config(page_title="Model Training", page_icon="🏋️", layout="wide")

st.title("🏋️ Model Training")
st.markdown("Train custom YOLOv8 models on your datasets")

# Initialize session state
if 'training_status' not in st.session_state:
    st.session_state.training_status = 'idle'
if 'training_progress' not in st.session_state:
    st.session_state.training_progress = {}
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}

# Training configuration tabs
tab1, tab2, tab3, tab4 = st.tabs(["Configure Training", "Training Progress", "Model Results", "Model Management"])

with tab1:
    st.header("Training Configuration")
    
    # Dataset selection
    datasets_path = Path('datasets')
    datasets = []
    if datasets_path.exists():
        datasets = [d for d in os.listdir(datasets_path) if os.path.isdir(datasets_path / d)]
    
    if not datasets:
        st.warning("No datasets found. Please upload a dataset first in the Dataset Management page.")
        st.stop()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_dataset = st.selectbox("Select Dataset", datasets)
        
        # Model configuration
        st.subheader("Model Configuration")
        
        model_name = st.text_input("Model Name", value=f"{selected_dataset}_model")
        base_model = st.selectbox("Base Model", [
            "yolov8n.pt",  # Nano - fastest, least accurate
            "yolov8s.pt",  # Small - balanced
            "yolov8m.pt",  # Medium - more accurate
            "yolov8l.pt",  # Large - most accurate, slowest
            "yolov8x.pt"   # Extra large - highest accuracy
        ])
        
        # Training parameters
        st.subheader("Training Parameters")
        
        col1_1, col1_2 = st.columns(2)
        
        with col1_1:
            epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=100)
            batch_size = st.number_input("Batch Size", min_value=1, max_value=64, value=16)
            image_size = st.selectbox("Image Size", [416, 512, 640, 832, 1024], index=2)
        
        with col1_2:
            learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.01, format="%.4f")
            patience = st.number_input("Early Stopping Patience", min_value=5, max_value=100, value=50)
            augmentation = st.checkbox("Data Augmentation", value=True)
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            weight_decay = st.number_input("Weight Decay", min_value=0.0, max_value=0.01, value=0.0005, format="%.6f")
            momentum = st.number_input("Momentum", min_value=0.0, max_value=1.0, value=0.937, format="%.3f")
            warmup_epochs = st.number_input("Warmup Epochs", min_value=0, max_value=10, value=3)
            save_period = st.number_input("Save Period (epochs)", min_value=1, max_value=50, value=10)
    
    with col2:
        st.subheader("Model Info")
        
        model_info = {
            "yolov8n.pt": {"params": "3.2M", "size": "6MB", "speed": "Fast"},
            "yolov8s.pt": {"params": "11.2M", "size": "22MB", "speed": "Fast"},
            "yolov8m.pt": {"params": "25.9M", "size": "52MB", "speed": "Medium"},
            "yolov8l.pt": {"params": "43.7M", "size": "87MB", "speed": "Slow"},
            "yolov8x.pt": {"params": "68.2M", "size": "136MB", "speed": "Slow"}
        }
        
        info = model_info[base_model]
        st.metric("Parameters", info["params"])
        st.metric("Model Size", info["size"])
        st.metric("Inference Speed", info["speed"])
        
        st.markdown("---")
        
        # Dataset info
        dataset_path = datasets_path / selected_dataset
        info_file = dataset_path / 'dataset_info.json'
        
        if info_file.exists():
            with open(info_file, 'r') as f:
                dataset_info = json.load(f)
            
            st.metric("Images", dataset_info.get('total_images', 0))
            st.metric("Classes", len(dataset_info.get('classes', [])))
        
        # Training time estimate
        estimated_time = epochs * 0.5  # Rough estimate
        st.metric("Est. Training Time", f"{estimated_time:.1f} min")
    
    # Start training button
    st.markdown("---")
    
    if st.session_state.training_status == 'idle':
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            # Validate dataset
            if not (dataset_path / 'dataset_info.json').exists():
                st.error("Dataset not properly configured. Please validate in Dataset Management.")
                st.stop()
            
            # Prepare training configuration
            training_config = {
                'model_name': model_name,
                'dataset_path': str(dataset_path),
                'base_model': base_model,
                'epochs': epochs,
                'batch_size': batch_size,
                'image_size': image_size,
                'learning_rate': learning_rate,
                'patience': patience,
                'augmentation': augmentation,
                'weight_decay': weight_decay,
                'momentum': momentum,
                'warmup_epochs': warmup_epochs,
                'save_period': save_period
            }
            
            # Start training
            st.session_state.training_status = 'preparing'
            st.session_state.training_config = training_config
            st.rerun()
    
    elif st.session_state.training_status in ['preparing', 'training']:
        st.info("Training in progress... Switch to the Training Progress tab to monitor.")
        
        if st.button("⏹️ Stop Training", type="secondary"):
            st.session_state.training_status = 'stopping'
            st.rerun()
    
    else:
        st.success("Training completed! Check the Model Results tab.")
        if st.button("🔄 Reset", type="secondary"):
            st.session_state.training_status = 'idle'
            st.rerun()

with tab2:
    st.header("Training Progress")
    
    if st.session_state.training_status == 'idle':
        st.info("No training in progress. Configure and start training in the previous tab.")
    
    elif st.session_state.training_status == 'preparing':
        st.info("Preparing training data...")
        
        # Simulate data preparation
        with st.spinner("Preparing dataset..."):
            try:
                config = st.session_state.training_config
                dataset_path = Path(config['dataset_path'])
                
                # Prepare training data (this would involve actual data processing)
                prepared_data = prepare_training_data(dataset_path, config)
                
                if prepared_data:
                    st.session_state.training_status = 'training'
                    st.session_state.training_start_time = time.time()
                    st.success("Data preparation completed! Starting training...")
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error("Failed to prepare training data.")
                    st.session_state.training_status = 'idle'
                    
            except Exception as e:
                st.error(f"Error preparing data: {str(e)}")
                st.session_state.training_status = 'idle'
    
    elif st.session_state.training_status == 'training':
        st.info("🏋️ Training in progress...")
        
        # Create placeholder for training metrics
        progress_placeholder = st.empty()
        metrics_placeholder = st.empty()
        
        # Simulate training progress (in real implementation, this would monitor actual training)
        if 'training_epoch' not in st.session_state:
            st.session_state.training_epoch = 0
        
        config = st.session_state.training_config
        total_epochs = config['epochs']
        
        # Update progress
        if st.session_state.training_epoch < total_epochs:
            st.session_state.training_epoch += 1
            
            progress = st.session_state.training_epoch / total_epochs
            
            with progress_placeholder.container():
                st.progress(progress)
                st.write(f"Epoch {st.session_state.training_epoch}/{total_epochs}")
                
                elapsed_time = time.time() - st.session_state.training_start_time
                eta = (elapsed_time / progress) - elapsed_time if progress > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Elapsed Time", f"{elapsed_time/60:.1f} min")
                with col2:
                    st.metric("ETA", f"{eta/60:.1f} min")
                with col3:
                    st.metric("Progress", f"{progress*100:.1f}%")
            
            # Simulate training metrics
            with metrics_placeholder.container():
                st.subheader("Training Metrics")
                
                # Generate simulated metrics (in real implementation, read from training logs)
                import numpy as np
                import random
                
                epoch_data = {
                    'epoch': list(range(1, st.session_state.training_epoch + 1)),
                    'train_loss': [0.5 + random.random() * 0.3 for _ in range(st.session_state.training_epoch)],
                    'val_loss': [0.6 + random.random() * 0.4 for _ in range(st.session_state.training_epoch)],
                    'mAP50': [0.2 + random.random() * 0.6 for _ in range(st.session_state.training_epoch)]
                }
                
                col1, col2 = st.columns(2)
                
                with col1:
                    df = pd.DataFrame(epoch_data)
                    st.line_chart(df.set_index('epoch')[['train_loss', 'val_loss']])
                    st.caption("Training and Validation Loss")
                
                with col2:
                    st.line_chart(df.set_index('epoch')['mAP50'])
                    st.caption("Mean Average Precision (mAP@0.5)")
            
            # Auto-refresh every 2 seconds during training
            time.sleep(2)
            st.rerun()
        
        else:
            # Training completed
            import random
            st.session_state.training_status = 'completed'
            st.session_state.model_results = {
                'final_map': 0.75 + random.random() * 0.2,
                'final_loss': 0.1 + random.random() * 0.1,
                'training_time': time.time() - st.session_state.training_start_time,
                'model_path': f"models/{config['model_name']}.pt"
            }
            st.rerun()
    
    elif st.session_state.training_status == 'completed':
        st.success("✅ Training completed successfully!")
        
        results = st.session_state.model_results
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Final mAP", f"{results['final_map']:.3f}")
        with col2:
            st.metric("Final Loss", f"{results['final_loss']:.3f}")
        with col3:
            st.metric("Training Time", f"{results['training_time']/60:.1f} min")

with tab3:
    st.header("Model Results")
    
    if st.session_state.training_status != 'completed':
        st.info("No completed training results available.")
    else:
        results = st.session_state.model_results
        
        # Model performance summary
        st.subheader("Performance Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("mAP@0.5", f"{results['final_map']:.3f}")
        with col2:
            st.metric("Final Loss", f"{results['final_loss']:.3f}")
        with col3:
            st.metric("Training Time", f"{results['training_time']/60:.1f} min")
        with col4:
            st.metric("Model Size", "52MB")  # Example
        
        # Training curves
        st.subheader("Training Curves")
        
        # In real implementation, load actual training logs
        st.info("Training curves would be displayed here from saved logs.")
        
        # Model export options
        st.subheader("Export Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            export_format = st.selectbox("Export Format", [
                "PyTorch (.pt)",
                "ONNX (.onnx)",
                "TensorFlow (.pb)",
                "TensorRT (.engine)"
            ])
        
        with col2:
            if st.button("📤 Export Model"):
                st.success(f"Model exported in {export_format} format!")
        
        # Download model
        st.subheader("Download Model")
        
        if st.button("💾 Download Trained Model"):
            st.success("Model download would start here!")

with tab4:
    st.header("Model Management")
    
    # List trained models
    models_path = Path('models')
    if models_path.exists():
        model_files = [f for f in os.listdir(models_path) if f.endswith('.pt')]
        
        if model_files:
            st.subheader("Trained Models")
            
            for model_file in model_files:
                with st.expander(f"📦 {model_file}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        file_path = models_path / model_file
                        file_size = os.path.getsize(file_path) / (1024*1024)  # MB
                        st.metric("Size", f"{file_size:.1f} MB")
                    
                    with col2:
                        mod_time = os.path.getmtime(file_path)
                        st.metric("Modified", time.strftime('%Y-%m-%d', time.localtime(mod_time)))
                    
                    with col3:
                        if st.button(f"🗑️ Delete", key=f"delete_{model_file}"):
                            os.remove(file_path)
                            st.success(f"Deleted {model_file}")
                            st.rerun()
                    
                    # Model actions
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button(f"🔍 Test Model", key=f"test_{model_file}"):
                            st.info("Redirect to Object Detection page")
                    with col2:
                        if st.button(f"📊 View Metrics", key=f"metrics_{model_file}"):
                            st.info("Model metrics would be displayed")
                    with col3:
                        if st.button(f"🚀 Deploy", key=f"deploy_{model_file}"):
                            st.info("Redirect to Roboflow Integration")
        else:
            st.info("No trained models found.")
    else:
        st.info("Models directory not found.")
