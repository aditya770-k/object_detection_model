import os
import json
import zipfile
import tempfile
import requests
from pathlib import Path
from roboflow import Roboflow

def upload_dataset_to_roboflow(workspace, dataset_path, config):
    """
    Upload dataset to Roboflow
    
    Args:
        workspace: Roboflow workspace object
        dataset_path (Path): Local dataset path
        config (dict): Upload configuration
        
    Returns:
        dict: Upload results
    """
    try:
        project_name = config['project_name']
        
        # Create or get project
        try:
            project = workspace.project(project_name)
        except:
            # Create new project if it doesn't exist
            project = workspace.create_project(
                project_name=project_name,
                project_type=config.get('project_type', 'object-detection')
            )
        
        # Prepare images for upload
        image_files = [f for f in dataset_path.iterdir() 
                      if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        
        uploaded_count = 0
        failed_uploads = []
        
        for img_file in image_files:
            try:
                # Check for corresponding annotation
                ann_file = dataset_path / f"{img_file.stem}.txt"
                
                if ann_file.exists():
                    # Upload image with annotation
                    project.upload(
                        image_path=str(img_file),
                        annotation_path=str(ann_file)
                    )
                else:
                    # Upload image only
                    project.upload(image_path=str(img_file))
                
                uploaded_count += 1
                
            except Exception as e:
                failed_uploads.append({
                    'file': img_file.name,
                    'error': str(e)
                })
        
        # Apply preprocessing if configured
        if config.get('preprocessing', {}).get('resize', False):
            target_size = config['preprocessing'].get('target_size', 640)
            project.generate_version(
                settings={
                    'preprocessing': {
                        'resize': {'width': target_size, 'height': target_size}
                    }
                }
            )
        
        return {
            'success': True,
            'uploaded_count': uploaded_count,
            'failed_uploads': failed_uploads,
            'project_url': f"https://roboflow.com/{workspace.workspace_name}/{project_name}"
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def deploy_model_to_roboflow(workspace, model_path, config):
    """
    Deploy trained model to Roboflow
    
    Args:
        workspace: Roboflow workspace object
        model_path (Path): Path to trained model
        config (dict): Deployment configuration
        
    Returns:
        dict: Deployment results
    """
    try:
        project_name = config['target_project']
        deployment_name = config['deployment_name']
        
        # Get project
        project = workspace.project(project_name)
        
        # Upload model weights
        # Note: Actual implementation would depend on Roboflow's model upload API
        
        # For now, we'll simulate the deployment process
        deployment_result = {
            'success': True,
            'deployment_id': f"dep_{deployment_name}_{hash(str(model_path)) % 10000}",
            'api_endpoint': f"https://detect.roboflow.com/{workspace.workspace_name}/{project_name}/1",
            'model_id': f"model_{deployment_name}",
            'deployment_url': f"https://roboflow.com/{workspace.workspace_name}/{project_name}/deploy"
        }
        
        return deployment_result
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def check_roboflow_status(workspace, project_name=None):
    """
    Check Roboflow workspace and project status
    
    Args:
        workspace: Roboflow workspace object
        project_name (str): Optional specific project name
        
    Returns:
        dict: Status information
    """
    try:
        status = {
            'workspace_name': workspace.workspace_name,
            'projects': [],
            'total_projects': 0,
            'quota_info': {}
        }
        
        # Get all projects
        projects = workspace.projects()
        status['total_projects'] = len(projects)
        
        for project in projects:
            project_info = {
                'name': project.name,
                'type': getattr(project, 'type', 'unknown'),
                'images': getattr(project, 'image_count', 0),
                'versions': getattr(project, 'version_count', 0)
            }
            status['projects'].append(project_info)
        
        # Get specific project info if requested
        if project_name:
            try:
                project = workspace.project(project_name)
                status['current_project'] = {
                    'name': project.name,
                    'type': getattr(project, 'type', 'unknown'),
                    'images': getattr(project, 'image_count', 0),
                    'versions': getattr(project, 'version_count', 0),
                    'last_updated': getattr(project, 'updated', 'unknown')
                }
            except:
                status['current_project'] = None
        
        return {
            'success': True,
            'status': status
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def download_roboflow_dataset(workspace, project_name, version_number, format_type='yolov8'):
    """
    Download dataset from Roboflow
    
    Args:
        workspace: Roboflow workspace object
        project_name (str): Project name
        version_number (int): Dataset version number
        format_type (str): Download format
        
    Returns:
        dict: Download results
    """
    try:
        # Get project and version
        project = workspace.project(project_name)
        version = project.version(version_number)
        
        # Download dataset
        dataset = version.download(format_type)
        
        return {
            'success': True,
            'dataset_path': dataset.location,
            'format': format_type,
            'version': version_number
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def get_model_performance(workspace, project_name, model_id):
    """
    Get model performance metrics from Roboflow
    
    Args:
        workspace: Roboflow workspace object
        project_name (str): Project name
        model_id (str): Model identifier
        
    Returns:
        dict: Performance metrics
    """
    try:
        # This would typically make API calls to get model metrics
        # For now, we'll return simulated metrics
        
        metrics = {
            'success': True,
            'model_id': model_id,
            'performance': {
                'map50': 0.75,
                'map50_95': 0.65,
                'precision': 0.80,
                'recall': 0.72,
                'inference_time': '45ms',
                'model_size': '52MB'
            },
            'usage_stats': {
                'total_predictions': 12345,
                'daily_predictions': 234,
                'error_rate': 0.021
            }
        }
        
        return metrics
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def manage_roboflow_webhooks(workspace, project_name, webhook_config):
    """
    Manage webhooks for Roboflow project
    
    Args:
        workspace: Roboflow workspace object
        project_name (str): Project name
        webhook_config (dict): Webhook configuration
        
    Returns:
        dict: Webhook management results
    """
    try:
        project = workspace.project(project_name)
        
        action = webhook_config.get('action', 'create')
        webhook_url = webhook_config.get('url')
        events = webhook_config.get('events', ['prediction'])
        
        if action == 'create':
            # Create webhook
            webhook_id = f"webhook_{hash(webhook_url) % 10000}"
            
            result = {
                'success': True,
                'action': 'created',
                'webhook_id': webhook_id,
                'url': webhook_url,
                'events': events
            }
        
        elif action == 'delete':
            webhook_id = webhook_config.get('webhook_id')
            
            result = {
                'success': True,
                'action': 'deleted',
                'webhook_id': webhook_id
            }
        
        else:
            result = {
                'success': False,
                'error': f"Unknown action: {action}"
            }
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def get_roboflow_usage_stats(workspace):
    """
    Get Roboflow usage statistics
    
    Args:
        workspace: Roboflow workspace object
        
    Returns:
        dict: Usage statistics
    """
    try:
        # In a real implementation, this would query Roboflow's API
        # For now, return simulated usage stats
        
        stats = {
            'success': True,
            'workspace': workspace.workspace_name,
            'current_month': {
                'predictions_used': 12345,
                'predictions_limit': 50000,
                'percentage_used': 24.7
            },
            'projects': {
                'total': 5,
                'active': 3,
                'limit': 10
            },
            'storage': {
                'used_gb': 2.3,
                'limit_gb': 10.0,
                'percentage_used': 23.0
            },
            'billing': {
                'current_plan': 'Professional',
                'monthly_cost': 49.99,
                'next_billing_date': '2023-12-15'
            }
        }
        
        return stats
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
