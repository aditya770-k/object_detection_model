import os
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import JSON
import streamlit as st

Base = declarative_base()

class Dataset(Base):
    __tablename__ = 'datasets'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False)
    description = Column(Text)
    total_images = Column(Integer, default=0)
    total_annotations = Column(Integer, default=0)
    classes = Column(JSON)
    class_distribution = Column(JSON)
    image_formats = Column(JSON)
    annotation_formats = Column(JSON)
    validation_passed = Column(Boolean, default=False)
    issues = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    training_jobs = relationship("TrainingJob", back_populates="dataset")

class TrainingJob(Base):
    __tablename__ = 'training_jobs'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(255), nullable=False)
    dataset_id = Column(Integer, ForeignKey('datasets.id'), nullable=False)
    base_model = Column(String(100), nullable=False)
    epochs = Column(Integer, nullable=False)
    batch_size = Column(Integer, nullable=False)
    image_size = Column(Integer, nullable=False)
    learning_rate = Column(Float, nullable=False)
    patience = Column(Integer, nullable=False)
    weight_decay = Column(Float, nullable=False)
    momentum = Column(Float, nullable=False)
    warmup_epochs = Column(Integer, nullable=False)
    save_period = Column(Integer, nullable=False)
    status = Column(String(50), default='pending')  # pending, training, completed, failed
    progress = Column(Float, default=0.0)  # 0.0 to 1.0
    current_epoch = Column(Integer, default=0)
    final_map = Column(Float)
    final_loss = Column(Float)
    training_time = Column(Float)  # in seconds
    model_path = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Relationships
    dataset = relationship("Dataset", back_populates="training_jobs")
    training_metrics = relationship("TrainingMetric", back_populates="training_job")

class TrainingMetric(Base):
    __tablename__ = 'training_metrics'
    
    id = Column(Integer, primary_key=True)
    training_job_id = Column(Integer, ForeignKey('training_jobs.id'), nullable=False)
    epoch = Column(Integer, nullable=False)
    train_loss = Column(Float)
    val_loss = Column(Float)
    map50 = Column(Float)
    map50_95 = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    training_job = relationship("TrainingJob", back_populates="training_metrics")

class Model(Base):
    __tablename__ = 'models'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    training_job_id = Column(Integer, ForeignKey('training_jobs.id'))
    model_path = Column(String(500), nullable=False)
    model_type = Column(String(100), default='yolov8')
    model_size_mb = Column(Float)
    final_map = Column(Float)
    final_loss = Column(Float)
    classes = Column(JSON)
    deployment_status = Column(String(50), default='local')  # local, deployed, failed
    roboflow_model_id = Column(String(255))
    roboflow_api_endpoint = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    detections = relationship("Detection", back_populates="model")

class Detection(Base):
    __tablename__ = 'detections'
    
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey('models.id'), nullable=False)
    image_name = Column(String(255), nullable=False)
    image_size = Column(JSON)  # {"width": 640, "height": 480}
    detections = Column(JSON)  # List of detection results
    confidence_threshold = Column(Float, nullable=False)
    iou_threshold = Column(Float, nullable=False)
    detection_count = Column(Integer, default=0)
    processing_time = Column(Float)  # in seconds
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    model = relationship("Model", back_populates="detections")

class RoboflowDeployment(Base):
    __tablename__ = 'roboflow_deployments'
    
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey('models.id'), nullable=False)
    project_name = Column(String(255), nullable=False)
    deployment_name = Column(String(255), nullable=False)
    api_endpoint = Column(String(500))
    deployment_id = Column(String(255))
    status = Column(String(50), default='pending')  # pending, active, failed
    confidence_threshold = Column(Float, nullable=False)
    overlap_threshold = Column(Float, nullable=False)
    max_objects = Column(Integer, nullable=False)
    enable_webhook = Column(Boolean, default=False)
    webhook_url = Column(String(500))
    enable_monitoring = Column(Boolean, default=True)
    enable_logging = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    deployed_at = Column(DateTime)

# Database connection and session management
@st.cache_resource
def get_database_engine():
    """Get database engine with connection pooling"""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")
    
    engine = create_engine(
        database_url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        echo=False
    )
    return engine

def get_session():
    """Get database session"""
    engine = get_database_engine()
    Session = sessionmaker(bind=engine)
    return Session()

def init_database():
    """Initialize database tables"""
    engine = get_database_engine()
    Base.metadata.create_all(engine)

def save_dataset_to_db(dataset_info):
    """Save dataset information to database"""
    session = get_session()
    try:
        # Check if dataset already exists
        existing = session.query(Dataset).filter_by(name=dataset_info['name']).first()
        if existing:
            # Update existing dataset
            for key, value in dataset_info.items():
                setattr(existing, key, value)
            existing.updated_at = datetime.utcnow()
            dataset = existing
        else:
            # Create new dataset
            dataset = Dataset(**dataset_info)
            session.add(dataset)
        
        session.commit()
        return dataset.id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_datasets_from_db():
    """Get all datasets from database"""
    session = get_session()
    try:
        datasets = session.query(Dataset).order_by(Dataset.created_at.desc()).all()
        return datasets
    finally:
        session.close()

def save_training_job_to_db(job_data):
    """Save training job to database"""
    session = get_session()
    try:
        training_job = TrainingJob(**job_data)
        session.add(training_job)
        session.commit()
        return training_job.id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def update_training_job_status(job_id, status, **kwargs):
    """Update training job status and metrics"""
    session = get_session()
    try:
        job = session.query(TrainingJob).filter_by(id=job_id).first()
        if job:
            job.status = status
            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            
            if status == 'training' and not job.started_at:
                job.started_at = datetime.utcnow()
            elif status == 'completed':
                job.completed_at = datetime.utcnow()
            
            session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def save_training_metric(job_id, epoch_data):
    """Save training metrics for an epoch"""
    session = get_session()
    try:
        metric = TrainingMetric(
            training_job_id=job_id,
            **epoch_data
        )
        session.add(metric)
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def save_model_to_db(model_data):
    """Save trained model information to database"""
    session = get_session()
    try:
        model = Model(**model_data)
        session.add(model)
        session.commit()
        return model.id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_models_from_db():
    """Get all models from database"""
    session = get_session()
    try:
        models = session.query(Model).order_by(Model.created_at.desc()).all()
        return models
    finally:
        session.close()

def save_detection_to_db(detection_data):
    """Save detection results to database"""
    session = get_session()
    try:
        detection = Detection(**detection_data)
        session.add(detection)
        session.commit()
        return detection.id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_detection_history(model_id=None, limit=100):
    """Get detection history"""
    session = get_session()
    try:
        query = session.query(Detection)
        if model_id:
            query = query.filter_by(model_id=model_id)
        detections = query.order_by(Detection.created_at.desc()).limit(limit).all()
        return detections
    finally:
        session.close()

def save_roboflow_deployment(deployment_data):
    """Save Roboflow deployment information"""
    session = get_session()
    try:
        deployment = RoboflowDeployment(**deployment_data)
        session.add(deployment)
        session.commit()
        return deployment.id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_training_history(dataset_id=None, model_name=None):
    """Get training history with metrics"""
    session = get_session()
    try:
        query = session.query(TrainingJob)
        if dataset_id:
            query = query.filter_by(dataset_id=dataset_id)
        if model_name:
            query = query.filter_by(model_name=model_name)
        
        jobs = query.order_by(TrainingJob.created_at.desc()).all()
        
        # Get metrics for each job
        for job in jobs:
            job.metrics = session.query(TrainingMetric).filter_by(
                training_job_id=job.id
            ).order_by(TrainingMetric.epoch).all()
        
        return jobs
    finally:
        session.close()

def get_database_stats():
    """Get database statistics for dashboard"""
    session = get_session()
    try:
        stats = {
            'total_datasets': session.query(Dataset).count(),
            'total_models': session.query(Model).count(),
            'total_training_jobs': session.query(TrainingJob).count(),
            'total_detections': session.query(Detection).count(),
            'active_deployments': session.query(RoboflowDeployment).filter_by(status='active').count()
        }
        return stats
    finally:
        session.close()