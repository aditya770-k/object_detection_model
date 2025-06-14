# Object Detection Trainer

## Overview

The Object Detection Trainer is a comprehensive Streamlit web application designed to provide an end-to-end workflow for custom object detection model development. The application enables users to upload datasets, train YOLOv8 models, perform object detection, and deploy models to Roboflow for production use.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit with multi-page navigation
- **Layout**: Wide layout with sidebar navigation and tabbed interfaces
- **Components**: 
  - Main dashboard (`app.py`)
  - Dataset Management page
  - Model Training page
  - Object Detection page
  - Roboflow Integration page

### Backend Architecture
- **Core Framework**: Python 3.11 with Streamlit
- **ML Framework**: Ultralytics YOLOv8 for object detection
- **Image Processing**: OpenCV, PIL, and NumPy for image manipulation
- **File Management**: Pathlib and standard library for dataset organization

### Deployment Strategy
- **Platform**: Replit with autoscale deployment
- **Container**: Nix-based environment with Python 3.11
- **Dependencies**: Graphics libraries (Cairo, FFmpeg, GTK3) for image processing
- **Port Configuration**: Streamlit server on port 5000

## Key Components

### 1. Dataset Management (`pages/1_Dataset_Management.py`)
- **Purpose**: Handle dataset upload, validation, and organization
- **Features**:
  - ZIP archive upload and extraction
  - Individual file uploads
  - Dataset validation and statistics
  - Export capabilities
- **Architecture**: Utility functions in `utils/dataset_utils.py` for dataset operations

### 2. Model Training (`pages/2_Model_Training.py`)
- **Purpose**: Configure and execute YOLOv8 model training
- **Features**:
  - Training configuration interface
  - Progress monitoring
  - Model results visualization
  - Model management
- **Architecture**: Training utilities in `utils/training_utils.py` with YOLO integration

### 3. Object Detection (`pages/3_Object_Detection.py`)
- **Purpose**: Perform inference using trained or pre-trained models
- **Features**:
  - Model selection (trained vs pre-trained)
  - Image detection
  - Camera detection
  - Batch processing
- **Architecture**: Detection utilities in `utils/detection_utils.py` for inference operations

### 4. Roboflow Integration (`pages/4_Roboflow_Integration.py`)
- **Purpose**: Deploy models and datasets to Roboflow platform
- **Features**:
  - API authentication
  - Dataset upload to Roboflow
  - Model deployment
  - Project monitoring
- **Architecture**: Roboflow utilities in `utils/roboflow_utils.py` for API interactions

## Data Flow

1. **Dataset Upload**: Users upload images and annotations via ZIP or individual files
2. **Dataset Validation**: System validates format, structure, and completeness
3. **Training Preparation**: Datasets are converted to YOLO format with YAML configuration
4. **Model Training**: YOLOv8 models are trained using prepared datasets
5. **Model Storage**: Trained models are saved in the `models/` directory
6. **Inference**: Models perform object detection on new images
7. **Roboflow Deployment**: Models and datasets can be deployed to Roboflow for production

## External Dependencies

### Core Dependencies
- **Streamlit**: Web application framework
- **Ultralytics**: YOLOv8 implementation for training and inference
- **OpenCV**: Computer vision operations
- **PIL**: Image processing
- **NumPy**: Numerical operations

### Integration Dependencies
- **Roboflow**: Cloud platform for model deployment
- **PyTorch**: Deep learning backend (CPU version configured)

### System Dependencies
- **Graphics Libraries**: Cairo, FFmpeg, GTK3 for advanced image processing
- **File Format Support**: JPEG, PNG, TIFF, WebP support

## Deployment Strategy

### Environment Configuration
- **Nix Package Manager**: Ensures reproducible environment
- **Python 3.11**: Latest stable Python version
- **Autoscale Deployment**: Automatic scaling based on demand

### File Structure
```
├── app.py                          # Main application entry point
├── pages/                          # Streamlit pages
│   ├── 1_Dataset_Management.py
│   ├── 2_Model_Training.py
│   ├── 3_Object_Detection.py
│   └── 4_Roboflow_Integration.py
├── utils/                          # Utility modules
│   ├── dataset_utils.py
│   ├── training_utils.py
│   ├── detection_utils.py
│   └── roboflow_utils.py
├── datasets/                       # Dataset storage
├── models/                         # Model storage
├── temp/                           # Temporary files
└── attached_assets/                # Documentation
```

### Resource Management
- **Memory**: Optimized for CPU-based PyTorch operations
- **Storage**: Local file system for datasets and models
- **Networking**: HTTP/HTTPS for Roboflow API integration

## Changelog
- June 14, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.