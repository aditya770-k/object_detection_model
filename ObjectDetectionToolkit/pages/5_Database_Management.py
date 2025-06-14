import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from utils.database import (
    get_database_stats, get_datasets_from_db, get_models_from_db, 
    get_training_history, get_detection_history, get_session
)
from utils.database import Dataset, TrainingJob, Model, Detection
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Database Management", page_icon="🗄️", layout="wide")

st.title("🗄️ Database Management")
st.markdown("Monitor and manage the application database")

# Database connection status
try:
    stats = get_database_stats()
    st.success("✅ Database connected successfully")
except Exception as e:
    st.error(f"❌ Database connection failed: {str(e)}")
    st.stop()

# Tabs for different database views
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Datasets", "Training History", "Detection History", "Analytics"])

with tab1:
    st.header("Database Overview")
    
    # Database statistics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Datasets", stats.get('total_datasets', 0))
    with col2:
        st.metric("Total Models", stats.get('total_models', 0))
    with col3:
        st.metric("Training Jobs", stats.get('total_training_jobs', 0))
    with col4:
        st.metric("Total Detections", stats.get('total_detections', 0))
    with col5:
        st.metric("Active Deployments", stats.get('active_deployments', 0))
    
    # Recent activity
    st.subheader("Recent Activity")
    
    try:
        session = get_session()
        
        # Recent datasets
        recent_datasets = session.query(Dataset).order_by(
            Dataset.created_at.desc()
        ).limit(5).all()
        
        # Recent training jobs
        recent_training = session.query(TrainingJob).order_by(
            TrainingJob.created_at.desc()
        ).limit(5).all()
        
        # Recent detections
        recent_detections = session.query(Detection).order_by(
            Detection.created_at.desc()
        ).limit(5).all()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Recent Datasets**")
            if recent_datasets:
                for ds in recent_datasets:
                    st.write(f"• {ds.name} ({ds.created_at.strftime('%Y-%m-%d %H:%M')})")
            else:
                st.info("No datasets found")
        
        with col2:
            st.markdown("**Recent Training Jobs**")
            if recent_training:
                for job in recent_training:
                    status_emoji = {"completed": "✅", "training": "🏋️", "failed": "❌", "pending": "⏳"}
                    emoji = status_emoji.get(job.status, "⏳")
                    st.write(f"• {emoji} {job.model_name} ({job.created_at.strftime('%Y-%m-%d %H:%M')})")
            else:
                st.info("No training jobs found")
        
        with col3:
            st.markdown("**Recent Detections**")
            if recent_detections:
                for det in recent_detections:
                    st.write(f"• {det.image_name} - {det.detection_count} objects ({det.created_at.strftime('%Y-%m-%d %H:%M')})")
            else:
                st.info("No detections found")
        
        session.close()
        
    except Exception as e:
        st.error(f"Error fetching recent activity: {str(e)}")

with tab2:
    st.header("Datasets in Database")
    
    try:
        datasets = get_datasets_from_db()
        
        if datasets:
            # Create dataset table
            dataset_data = []
            for ds in datasets:
                dataset_data.append({
                    'Name': ds.name,
                    'Description': ds.description[:50] + '...' if ds.description and len(ds.description) > 50 else ds.description or '',
                    'Images': ds.total_images,
                    'Annotations': ds.total_annotations,
                    'Classes': len(ds.classes) if ds.classes else 0,
                    'Validated': '✅' if ds.validation_passed else '❌',
                    'Created': ds.created_at.strftime('%Y-%m-%d %H:%M')
                })
            
            df = pd.DataFrame(dataset_data)
            st.dataframe(df, use_container_width=True)
            
            # Dataset details
            if st.session_state.get('selected_dataset_detail'):
                selected_name = st.session_state.selected_dataset_detail
                selected_ds = next((ds for ds in datasets if ds.name == selected_name), None)
                
                if selected_ds:
                    with st.expander(f"Details: {selected_ds.name}", expanded=True):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Description:** {selected_ds.description}")
                            st.write(f"**Total Images:** {selected_ds.total_images}")
                            st.write(f"**Total Annotations:** {selected_ds.total_annotations}")
                            st.write(f"**Validation Status:** {'Passed' if selected_ds.validation_passed else 'Failed'}")
                        
                        with col2:
                            if selected_ds.class_distribution:
                                st.write("**Class Distribution:**")
                                class_df = pd.DataFrame(
                                    list(selected_ds.class_distribution.items()),
                                    columns=['Class', 'Count']
                                )
                                st.bar_chart(class_df.set_index('Class'))
                        
                        if selected_ds.issues:
                            st.write("**Issues:**")
                            for issue in selected_ds.issues:
                                st.warning(f"• {issue}")
            
            # Select dataset for details
            dataset_names = [ds.name for ds in datasets]
            selected = st.selectbox("Select dataset for details", dataset_names, key="dataset_detail_select")
            if st.button("Show Details"):
                st.session_state.selected_dataset_detail = selected
                st.rerun()
                
        else:
            st.info("No datasets found in database")
            
    except Exception as e:
        st.error(f"Error loading datasets: {str(e)}")

with tab3:
    st.header("Training History")
    
    try:
        training_jobs = get_training_history()
        
        if training_jobs:
            # Training jobs table
            training_data = []
            for job in training_jobs:
                duration = None
                if job.completed_at and job.started_at:
                    duration = (job.completed_at - job.started_at).total_seconds() / 60  # minutes
                elif job.training_time:
                    duration = job.training_time / 60
                
                training_data.append({
                    'Model Name': job.model_name,
                    'Status': job.status.title(),
                    'Dataset': job.dataset.name if job.dataset else 'Unknown',
                    'Base Model': job.base_model,
                    'Epochs': job.epochs,
                    'Final mAP': f"{job.final_map:.3f}" if job.final_map else 'N/A',
                    'Final Loss': f"{job.final_loss:.3f}" if job.final_loss else 'N/A',
                    'Duration (min)': f"{duration:.1f}" if duration else 'N/A',
                    'Created': job.created_at.strftime('%Y-%m-%d %H:%M')
                })
            
            df = pd.DataFrame(training_data)
            st.dataframe(df, use_container_width=True)
            
            # Training metrics visualization
            if len(training_jobs) > 0:
                st.subheader("Training Performance Metrics")
                
                completed_jobs = [job for job in training_jobs if job.status == 'completed' and job.final_map]
                
                if completed_jobs:
                    # mAP comparison
                    fig_map = px.bar(
                        x=[job.model_name for job in completed_jobs],
                        y=[job.final_map for job in completed_jobs],
                        title="Final mAP by Model",
                        labels={'x': 'Model Name', 'y': 'mAP@0.5'}
                    )
                    st.plotly_chart(fig_map, use_container_width=True)
                    
                    # Training time vs performance
                    if any(job.training_time for job in completed_jobs):
                        fig_scatter = px.scatter(
                            x=[job.training_time/60 for job in completed_jobs if job.training_time],
                            y=[job.final_map for job in completed_jobs if job.training_time],
                            text=[job.model_name for job in completed_jobs if job.training_time],
                            title="Training Time vs Performance",
                            labels={'x': 'Training Time (minutes)', 'y': 'mAP@0.5'}
                        )
                        fig_scatter.update_traces(textposition="top center")
                        st.plotly_chart(fig_scatter, use_container_width=True)
                
        else:
            st.info("No training jobs found in database")
            
    except Exception as e:
        st.error(f"Error loading training history: {str(e)}")

with tab4:
    st.header("Detection History")
    
    try:
        detections = get_detection_history(limit=100)
        
        if detections:
            # Detection history table
            detection_data = []
            for det in detections:
                detection_data.append({
                    'Image': det.image_name,
                    'Model': det.model.name if det.model else 'Unknown',
                    'Detections': det.detection_count,
                    'Confidence': det.confidence_threshold,
                    'IoU': det.iou_threshold,
                    'Processing Time (s)': f"{det.processing_time:.3f}" if det.processing_time else 'N/A',
                    'Timestamp': det.created_at.strftime('%Y-%m-%d %H:%M:%S')
                })
            
            df = pd.DataFrame(detection_data)
            st.dataframe(df, use_container_width=True)
            
            # Detection analytics
            st.subheader("Detection Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Detections over time
                df['Date'] = pd.to_datetime([det.created_at for det in detections])
                daily_detections = df.groupby(df['Date'].dt.date).size().reset_index()
                daily_detections.columns = ['Date', 'Count']
                
                fig_time = px.line(
                    daily_detections, 
                    x='Date', 
                    y='Count',
                    title="Detections Over Time"
                )
                st.plotly_chart(fig_time, use_container_width=True)
            
            with col2:
                # Average detections per image by model
                model_avg = df.groupby('Model')['Detections'].mean().reset_index()
                
                fig_model = px.bar(
                    model_avg,
                    x='Model',
                    y='Detections',
                    title="Average Detections per Image by Model"
                )
                st.plotly_chart(fig_model, use_container_width=True)
                
        else:
            st.info("No detections found in database")
            
    except Exception as e:
        st.error(f"Error loading detection history: {str(e)}")

with tab5:
    st.header("Analytics Dashboard")
    
    try:
        # Get data for analytics
        datasets = get_datasets_from_db()
        training_jobs = get_training_history()
        detections = get_detection_history(limit=1000)
        
        # Usage statistics
        st.subheader("Usage Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Dataset creation over time
            if datasets:
                dataset_dates = [ds.created_at.date() for ds in datasets]
                dataset_df = pd.DataFrame(dataset_dates, columns=['Date'])
                dataset_counts = dataset_df.groupby('Date').size().reset_index()
                dataset_counts.columns = ['Date', 'Datasets Created']
                
                fig = px.line(dataset_counts, x='Date', y='Datasets Created', title="Datasets Created Over Time")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Training job status distribution
            if training_jobs:
                status_counts = {}
                for job in training_jobs:
                    status_counts[job.status] = status_counts.get(job.status, 0) + 1
                
                fig = px.pie(
                    values=list(status_counts.values()),
                    names=list(status_counts.keys()),
                    title="Training Job Status Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # Detection activity heatmap
            if detections:
                detection_df = pd.DataFrame([{
                    'hour': det.created_at.hour,
                    'day': det.created_at.strftime('%A'),
                    'count': 1
                } for det in detections])
                
                heatmap_data = detection_df.groupby(['day', 'hour']).sum().reset_index()
                
                fig = px.density_heatmap(
                    heatmap_data,
                    x='hour',
                    y='day',
                    z='count',
                    title="Detection Activity Heatmap"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics
        st.subheader("Performance Summary")
        
        if training_jobs:
            completed_jobs = [job for job in training_jobs if job.status == 'completed']
            
            if completed_jobs:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_map = sum(job.final_map for job in completed_jobs if job.final_map) / len([job for job in completed_jobs if job.final_map])
                    st.metric("Average mAP", f"{avg_map:.3f}")
                
                with col2:
                    avg_time = sum(job.training_time for job in completed_jobs if job.training_time) / len([job for job in completed_jobs if job.training_time])
                    st.metric("Avg Training Time", f"{avg_time/60:.1f} min")
                
                with col3:
                    success_rate = len(completed_jobs) / len(training_jobs) * 100
                    st.metric("Training Success Rate", f"{success_rate:.1f}%")
                
                with col4:
                    if detections:
                        avg_detections = sum(det.detection_count for det in detections) / len(detections)
                        st.metric("Avg Detections/Image", f"{avg_detections:.1f}")
        
    except Exception as e:
        st.error(f"Error loading analytics: {str(e)}")

# Database maintenance
st.markdown("---")
st.subheader("Database Maintenance")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔄 Refresh Stats"):
        st.cache_resource.clear()
        st.rerun()

with col2:
    if st.button("📊 Export Data"):
        st.info("Data export functionality would be implemented here")

with col3:
    if st.button("🧹 Cleanup Old Records"):
        st.info("Cleanup functionality would be implemented here")