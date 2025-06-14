import streamlit as st
import os
import json
import requests
from pathlib import Path
import zipfile
import tempfile
from roboflow import Roboflow
from utils.roboflow_utils import upload_dataset_to_roboflow, deploy_model_to_roboflow, check_roboflow_status

st.set_page_config(page_title="Roboflow Integration", page_icon="🚀", layout="wide")

st.title("🚀 Roboflow Integration")
st.markdown("Deploy your models and datasets to Roboflow for production use")

# Initialize session state
if 'roboflow_client' not in st.session_state:
    st.session_state.roboflow_client = None
if 'roboflow_projects' not in st.session_state:
    st.session_state.roboflow_projects = []

# Roboflow integration tabs
tab1, tab2, tab3, tab4 = st.tabs(["Setup & Auth", "Dataset Upload", "Model Deployment", "Monitor & Manage"])

with tab1:
    st.header("Roboflow Setup & Authentication")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("API Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Roboflow API Key",
            type="password",
            value=os.getenv("ROBOFLOW_API_KEY", ""),
            help="Get your API key from https://roboflow.com/settings/api"
        )
        
        workspace = st.text_input(
            "Workspace Name",
            value=os.getenv("ROBOFLOW_WORKSPACE", ""),
            help="Your Roboflow workspace identifier"
        )
        
        # Test connection
        if st.button("🔐 Test Connection"):
            if api_key and workspace:
                with st.spinner("Testing Roboflow connection..."):
                    try:
                        rf = Roboflow(api_key=api_key)
                        ws = rf.workspace(workspace)
                        
                        st.session_state.roboflow_client = rf
                        st.session_state.roboflow_workspace = ws
                        
                        # Get projects
                        projects = ws.projects()
                        st.session_state.roboflow_projects = [p.name for p in projects]
                        
                        st.success("✅ Connected to Roboflow successfully!")
                        st.info(f"Found {len(projects)} projects in workspace '{workspace}'")
                        
                    except Exception as e:
                        st.error(f"❌ Connection failed: {str(e)}")
                        st.info("Please check your API key and workspace name")
            else:
                st.warning("Please enter both API key and workspace name")
        
        # Save credentials
        if st.session_state.roboflow_client is not None:
            if st.button("💾 Save Credentials"):
                # In a real app, you'd save these securely
                st.success("Credentials saved for this session")
    
    with col2:
        st.subheader("Connection Status")
        
        if st.session_state.roboflow_client is not None:
            st.success("🟢 Connected")
            st.metric("Workspace", workspace)
            st.metric("Projects", len(st.session_state.roboflow_projects))
        else:
            st.error("🔴 Not Connected")
            st.info("Enter API credentials and test connection")
        
        # Quick links
        st.subheader("Quick Links")
        st.markdown("""
        - [Get API Key](https://roboflow.com/settings/api)
        - [Roboflow Dashboard](https://roboflow.com/)
        - [Documentation](https://docs.roboflow.com/)
        - [Pricing](https://roboflow.com/pricing)
        """)

with tab2:
    st.header("Dataset Upload to Roboflow")
    
    if st.session_state.roboflow_client is None:
        st.warning("Please connect to Roboflow first in the Setup & Auth tab.")
        st.stop()
    
    # Dataset selection
    datasets_path = Path('datasets')
    datasets = []
    if datasets_path.exists():
        datasets = [d for d in os.listdir(datasets_path) if os.path.isdir(datasets_path / d)]
    
    if not datasets:
        st.warning("No datasets found. Please create a dataset first.")
        st.stop()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_dataset = st.selectbox("Select Dataset to Upload", datasets)
        
        # Project configuration
        st.subheader("Project Configuration")
        
        project_action = st.radio("Project Action", ["Create New Project", "Upload to Existing Project"])
        
        if project_action == "Create New Project":
            project_name = st.text_input("New Project Name", value=f"{selected_dataset}_project")
            project_type = st.selectbox("Project Type", ["Object Detection", "Classification", "Segmentation"])
        else:
            if st.session_state.roboflow_projects:
                project_name = st.selectbox("Existing Project", st.session_state.roboflow_projects)
            else:
                st.warning("No existing projects found in workspace")
                st.stop()
        
        # Upload settings
        st.subheader("Upload Settings")
        
        col1_1, col1_2 = st.columns(2)
        
        with col1_1:
            split_data = st.checkbox("Split Dataset", value=True)
            if split_data:
                train_split = st.slider("Training Split", 0.5, 0.9, 0.8)
                val_split = st.slider("Validation Split", 0.05, 0.3, 0.15)
                test_split = 1.0 - train_split - val_split
                st.write(f"Test Split: {test_split:.2f}")
        
        with col1_2:
            apply_preprocessing = st.checkbox("Apply Preprocessing", value=False)
            if apply_preprocessing:
                resize_images = st.checkbox("Resize Images", value=True)
                if resize_images:
                    target_size = st.selectbox("Target Size", [416, 512, 640, 832])
                
                auto_contrast = st.checkbox("Auto Contrast", value=False)
        
        # Upload button
        if st.button("📤 Upload Dataset to Roboflow", type="primary"):
            dataset_path = datasets_path / selected_dataset
            
            with st.spinner("Uploading dataset to Roboflow..."):
                try:
                    # Load dataset info
                    info_file = dataset_path / 'dataset_info.json'
                    if info_file.exists():
                        with open(info_file, 'r') as f:
                            dataset_info = json.load(f)
                    else:
                        st.error("Dataset info not found. Please validate dataset first.")
                        st.stop()
                    
                    # Configure upload parameters
                    upload_config = {
                        'project_name': project_name,
                        'project_type': project_type,
                        'split_data': split_data,
                        'train_split': train_split if split_data else 0.8,
                        'val_split': val_split if split_data else 0.15,
                        'preprocessing': {
                            'resize': resize_images if apply_preprocessing else False,
                            'target_size': target_size if apply_preprocessing and resize_images else 640,
                            'auto_contrast': auto_contrast if apply_preprocessing else False
                        }
                    }
                    
                    # Upload dataset
                    result = upload_dataset_to_roboflow(
                        st.session_state.roboflow_workspace,
                        dataset_path,
                        upload_config
                    )
                    
                    if result['success']:
                        st.success("✅ Dataset uploaded successfully!")
                        st.info(f"Project URL: {result.get('project_url', 'N/A')}")
                        
                        # Update projects list
                        projects = st.session_state.roboflow_workspace.projects()
                        st.session_state.roboflow_projects = [p.name for p in projects]
                    else:
                        st.error(f"❌ Upload failed: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    st.error(f"Error uploading dataset: {str(e)}")
    
    with col2:
        st.subheader("Dataset Preview")
        
        if selected_dataset:
            dataset_path = datasets_path / selected_dataset
            info_file = dataset_path / 'dataset_info.json'
            
            if info_file.exists():
                with open(info_file, 'r') as f:
                    dataset_info = json.load(f)
                
                st.metric("Images", dataset_info.get('total_images', 0))
                st.metric("Annotations", dataset_info.get('total_annotations', 0))
                st.metric("Classes", len(dataset_info.get('classes', [])))
                
                # Class list
                if 'classes' in dataset_info:
                    st.subheader("Classes")
                    for i, class_name in enumerate(dataset_info['classes']):
                        st.write(f"{i}: {class_name}")
        
        st.subheader("Upload Guidelines")
        st.markdown("""
        **Supported Formats:**
        - Images: JPG, PNG
        - Annotations: YOLO, COCO, Pascal VOC
        
        **Best Practices:**
        - Validate dataset before upload
        - Use descriptive project names
        - Apply preprocessing if needed
        - Check quota limits
        """)

with tab3:
    st.header("Model Deployment")
    
    if st.session_state.roboflow_client is None:
        st.warning("Please connect to Roboflow first in the Setup & Auth tab.")
        st.stop()
    
    # Model selection
    models_path = Path('models')
    model_files = []
    if models_path.exists():
        model_files = [f for f in os.listdir(models_path) if f.endswith('.pt')]
    
    if not model_files:
        st.warning("No trained models found. Train a model first.")
        st.stop()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_model = st.selectbox("Select Model to Deploy", model_files)
        
        # Deployment configuration
        st.subheader("Deployment Configuration")
        
        if st.session_state.roboflow_projects:
            target_project = st.selectbox("Target Project", st.session_state.roboflow_projects)
        else:
            st.warning("No projects found. Upload a dataset first.")
            st.stop()
        
        deployment_name = st.text_input("Deployment Name", value=f"{selected_model.replace('.pt', '')}_deployment")
        
        # Model settings
        st.subheader("Model Settings")
        
        col1_1, col1_2 = st.columns(2)
        
        with col1_1:
            confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.5)
            overlap_threshold = st.slider("Overlap Threshold", 0.1, 0.9, 0.5)
        
        with col1_2:
            max_objects = st.number_input("Max Objects", 1, 100, 25)
            deployment_type = st.selectbox("Deployment Type", ["Hosted API", "Edge Deployment"])
        
        # Deployment options
        st.subheader("Deployment Options")
        
        enable_webhook = st.checkbox("Enable Webhooks")
        if enable_webhook:
            webhook_url = st.text_input("Webhook URL")
        
        enable_monitoring = st.checkbox("Enable Performance Monitoring", value=True)
        enable_logging = st.checkbox("Enable Request Logging", value=True)
        
        # Deploy button
        if st.button("🚀 Deploy Model", type="primary"):
            model_path = models_path / selected_model
            
            with st.spinner("Deploying model to Roboflow..."):
                try:
                    # Configure deployment
                    deploy_config = {
                        'deployment_name': deployment_name,
                        'target_project': target_project,
                        'confidence_threshold': confidence_threshold,
                        'overlap_threshold': overlap_threshold,
                        'max_objects': max_objects,
                        'deployment_type': deployment_type,
                        'enable_webhook': enable_webhook,
                        'webhook_url': webhook_url if enable_webhook else None,
                        'enable_monitoring': enable_monitoring,
                        'enable_logging': enable_logging
                    }
                    
                    # Deploy model
                    result = deploy_model_to_roboflow(
                        st.session_state.roboflow_workspace,
                        model_path,
                        deploy_config
                    )
                    
                    if result['success']:
                        st.success("✅ Model deployed successfully!")
                        st.info(f"API Endpoint: {result.get('api_endpoint', 'N/A')}")
                        st.info(f"Model ID: {result.get('model_id', 'N/A')}")
                        
                        # Show API usage example
                        with st.expander("API Usage Example"):
                            st.code(f"""
import requests

# Example API call
url = "{result.get('api_endpoint', 'YOUR_API_ENDPOINT')}"
headers = {{"Authorization": "Bearer YOUR_API_KEY"}}

# Upload image for inference
with open("image.jpg", "rb") as f:
    response = requests.post(url, files={{"file": f}}, headers=headers)
    
predictions = response.json()
print(predictions)
                            """, language="python")
                    else:
                        st.error(f"❌ Deployment failed: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    st.error(f"Error deploying model: {str(e)}")
    
    with col2:
        st.subheader("Model Info")
        
        if selected_model:
            model_path = models_path / selected_model
            file_size = os.path.getsize(model_path) / (1024*1024)  # MB
            
            st.metric("Model File", selected_model)
            st.metric("File Size", f"{file_size:.1f} MB")
            st.metric("Format", "PyTorch")
        
        st.subheader("Deployment Types")
        st.markdown("""
        **Hosted API:**
        - Cloud-based inference
        - Automatic scaling
        - Pay per prediction
        
        **Edge Deployment:**
        - Local inference
        - Reduced latency
        - Offline capability
        """)

with tab4:
    st.header("Monitor & Manage Deployments")
    
    if st.session_state.roboflow_client is None:
        st.warning("Please connect to Roboflow first in the Setup & Auth tab.")
        st.stop()
    
    # Deployment monitoring
    st.subheader("Active Deployments")
    
    # This would fetch actual deployments from Roboflow
    if st.button("🔄 Refresh Deployments"):
        with st.spinner("Fetching deployment status..."):
            try:
                # Mock deployment data (in real implementation, fetch from Roboflow API)
                deployments = [
                    {
                        'name': 'my_model_deployment',
                        'status': 'Active',
                        'project': 'my_project',
                        'requests_today': 1234,
                        'avg_latency': '45ms',
                        'accuracy': '0.87'
                    }
                ]
                
                for deployment in deployments:
                    with st.expander(f"📡 {deployment['name']}"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Status", deployment['status'])
                            st.metric("Project", deployment['project'])
                        
                        with col2:
                            st.metric("Requests Today", deployment['requests_today'])
                            st.metric("Avg Latency", deployment['avg_latency'])
                        
                        with col3:
                            st.metric("Accuracy", deployment['accuracy'])
                        
                        # Deployment actions
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            if st.button("📊 View Analytics", key=f"analytics_{deployment['name']}"):
                                st.info("Would open analytics dashboard")
                        
                        with col2:
                            if st.button("⚙️ Configure", key=f"config_{deployment['name']}"):
                                st.info("Would open configuration panel")
                        
                        with col3:
                            if st.button("🧪 Test API", key=f"test_{deployment['name']}"):
                                st.info("Would open API testing interface")
                        
                        with col4:
                            if st.button("🗑️ Delete", key=f"delete_{deployment['name']}"):
                                st.warning("Deployment deletion would be confirmed here")
                
                if not deployments:
                    st.info("No active deployments found.")
                    
            except Exception as e:
                st.error(f"Error fetching deployments: {str(e)}")
    
    # Usage statistics
    st.subheader("Usage Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Requests", "12,345")
    with col2:
        st.metric("Successful Predictions", "12,089")
    with col3:
        st.metric("Error Rate", "2.1%")
    with col4:
        st.metric("Avg Response Time", "67ms")
    
    # Billing information
    st.subheader("Billing & Usage")
    
    with st.expander("Billing Information"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Current Plan", "Professional")
            st.metric("Monthly Quota", "50,000 predictions")
            st.metric("Used This Month", "12,345 (24.7%)")
        
        with col2:
            st.metric("Additional Predictions", "$0.001 each")
            st.metric("Estimated Monthly Cost", "$15.23")
            st.metric("Next Billing Date", "Dec 15, 2023")
    
    # Support and documentation
    st.subheader("Support & Resources")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Documentation:**
        - [API Reference](https://docs.roboflow.com/api)
        - [SDKs & Libraries](https://docs.roboflow.com/sdks)
        - [Deployment Guide](https://docs.roboflow.com/deploy)
        """)
    
    with col2:
        st.markdown("""
        **Support:**
        - [Community Forum](https://community.roboflow.com/)
        - [Contact Support](https://roboflow.com/support)
        - [Status Page](https://status.roboflow.com/)
        """)
