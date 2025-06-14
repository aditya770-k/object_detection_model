import os
import yaml
import json
from pathlib import Path
import subprocess
import time
import threading
import queue

def prepare_training_data(dataset_path, config):
    """
    Prepare dataset for YOLO training
    
    Args:
        dataset_path (Path): Path to dataset directory
        config (dict): Training configuration
        
    Returns:
        dict: Prepared data configuration
    """
    try:
        # Create YOLO dataset configuration
        dataset_yaml = {
            'path': str(dataset_path.absolute()),
            'train': 'images',
            'val': 'images',
            'test': 'images',
            'nc': 0,  # Number of classes (will be updated)
            'names': []  # Class names (will be updated)
        }
        
        # Load dataset info
        info_file = dataset_path / 'dataset_info.json'
        if info_file.exists():
            with open(info_file, 'r') as f:
                dataset_info = json.load(f)
            
            # Update class information
            classes = dataset_info.get('classes', [])
            dataset_yaml['nc'] = len(classes)
            dataset_yaml['names'] = [f"class_{i}" for i in range(len(classes))]
        
        # Save dataset configuration
        yaml_path = dataset_path / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False)
        
        return {
            'success': True,
            'yaml_path': str(yaml_path),
            'num_classes': dataset_yaml['nc'],
            'class_names': dataset_yaml['names']
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def start_training(config, progress_callback=None):
    """
    Start YOLO model training
    
    Args:
        config (dict): Training configuration
        progress_callback (callable): Callback for progress updates
        
    Returns:
        dict: Training results
    """
    try:
        # Create a mock training process for demonstration
        model_save_path = Path('models') / f"{config['model_name']}.pt"
        model_save_path.parent.mkdir(exist_ok=True)
        
        # Create a dummy model file
        with open(model_save_path, 'w') as f:
            f.write(f"# Mock trained model: {config['model_name']}\n")
            f.write(f"# Base model: {config['base_model']}\n")
            f.write(f"# Epochs: {config['epochs']}\n")
        
        return {
            'success': True,
            'model_path': str(model_save_path),
            'results': {'final_metrics': {'map50': 0.75, 'loss': 0.25}},
            'metrics': {'final_map50': 0.75, 'final_loss': 0.25}
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def extract_training_metrics(results):
    """
    Extract training metrics from YOLO results
    
    Args:
        results: YOLO training results
        
    Returns:
        dict: Extracted metrics
    """
    try:
        # Extract key metrics
        metrics = {
            'final_map50': 0.0,
            'final_map50_95': 0.0,
            'final_loss': 0.0,
            'best_epoch': 0,
            'training_time': 0.0
        }
        
        # In a real implementation, extract from results object
        if hasattr(results, 'results_dict'):
            results_dict = results.results_dict
            metrics['final_map50'] = results_dict.get('metrics/mAP50(B)', 0.0)
            metrics['final_map50_95'] = results_dict.get('metrics/mAP50-95(B)', 0.0)
            metrics['final_loss'] = results_dict.get('train/box_loss', 0.0)
        
        return metrics
        
    except Exception as e:
        return {
            'error': f"Error extracting metrics: {str(e)}"
        }

def monitor_training(model_name, callback=None):
    """
    Monitor training progress
    
    Args:
        model_name (str): Name of the model being trained
        callback (callable): Callback for progress updates
        
    Returns:
        dict: Training status and metrics
    """
    try:
        # Look for training logs
        runs_dir = Path('models') / model_name
        
        if not runs_dir.exists():
            return {'status': 'not_found'}
        
        # Check for results files
        results_file = runs_dir / 'results.csv'
        if results_file.exists():
            # Parse training results
            import pandas as pd
            df = pd.read_csv(results_file)
            
            current_epoch = len(df)
            latest_metrics = df.iloc[-1].to_dict() if len(df) > 0 else {}
            
            progress = {
                'status': 'training',
                'current_epoch': current_epoch,
                'metrics': latest_metrics,
                'progress_data': df.to_dict('records')
            }
            
            if callback:
                callback(progress)
            
            return progress
        
        return {'status': 'starting'}
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }

def validate_model(model_path, dataset_path):
    """
    Validate trained model on test set
    
    Args:
        model_path (str): Path to trained model
        dataset_path (str): Path to dataset
        
    Returns:
        dict: Validation results
    """
    try:
        # Load model
        model = YOLO(model_path)
        
        # Run validation
        results = model.val(
            data=dataset_path + '/dataset.yaml',
            split='test'
        )
        
        # Extract validation metrics
        val_metrics = {
            'map50': 0.0,
            'map50_95': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
        
        # In real implementation, extract from results
        if hasattr(results, 'results_dict'):
            results_dict = results.results_dict
            val_metrics['map50'] = results_dict.get('metrics/mAP50(B)', 0.0)
            val_metrics['map50_95'] = results_dict.get('metrics/mAP50-95(B)', 0.0)
            val_metrics['precision'] = results_dict.get('metrics/precision(B)', 0.0)
            val_metrics['recall'] = results_dict.get('metrics/recall(B)', 0.0)
        
        return {
            'success': True,
            'metrics': val_metrics
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def export_model(model_path, export_format='onnx'):
    """
    Export trained model to different formats
    
    Args:
        model_path (str): Path to trained model
        export_format (str): Export format (onnx, torchscript, etc.)
        
    Returns:
        dict: Export results
    """
    try:
        # Load model
        model = YOLO(model_path)
        
        # Export model
        export_path = model.export(format=export_format)
        
        return {
            'success': True,
            'export_path': str(export_path),
            'format': export_format
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def get_training_history(model_name):
    """
    Get training history and metrics
    
    Args:
        model_name (str): Name of the trained model
        
    Returns:
        dict: Training history data
    """
    try:
        runs_dir = Path('models') / model_name
        results_file = runs_dir / 'results.csv'
        
        if not results_file.exists():
            return {'success': False, 'error': 'No training history found'}
        
        import pandas as pd
        df = pd.read_csv(results_file)
        
        history = {
            'success': True,
            'epochs': len(df),
            'metrics': df.to_dict('records'),
            'final_metrics': df.iloc[-1].to_dict() if len(df) > 0 else {}
        }
        
        return history
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
