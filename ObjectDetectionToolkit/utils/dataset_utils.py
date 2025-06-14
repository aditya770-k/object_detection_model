import os
import json
import shutil
from pathlib import Path
from collections import Counter
from PIL import Image
import xml.etree.ElementTree as ET

def validate_dataset(dataset_path):
    """
    Validate a dataset for object detection training
    
    Args:
        dataset_path (Path): Path to the dataset directory
        
    Returns:
        dict: Validation results with statistics and issues
    """
    results = {
        'valid': True,
        'stats': {
            'total_images': 0,
            'total_annotations': 0,
            'image_formats': {},
            'annotation_formats': {},
            'classes': []
        },
        'issues': []
    }
    
    try:
        # Check if directory exists
        if not dataset_path.exists():
            results['valid'] = False
            results['issues'].append("Dataset directory does not exist")
            return results
        
        # Get all files
        all_files = list(dataset_path.iterdir())
        
        # Separate images and annotations
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        annotation_extensions = {'.txt', '.json', '.xml'}
        
        image_files = []
        annotation_files = []
        
        for file in all_files:
            if file.suffix.lower() in image_extensions:
                image_files.append(file)
            elif file.suffix.lower() in annotation_extensions:
                annotation_files.append(file)
        
        # Update stats
        results['stats']['total_images'] = len(image_files)
        results['stats']['total_annotations'] = len(annotation_files)
        
        # Count formats
        for img_file in image_files:
            ext = img_file.suffix.lower()
            results['stats']['image_formats'][ext] = results['stats']['image_formats'].get(ext, 0) + 1
        
        for ann_file in annotation_files:
            ext = ann_file.suffix.lower()
            results['stats']['annotation_formats'][ext] = results['stats']['annotation_formats'].get(ext, 0) + 1
        
        # Validate images
        corrupted_images = []
        for img_file in image_files:
            try:
                with Image.open(img_file) as img:
                    img.verify()
            except Exception:
                corrupted_images.append(img_file.name)
        
        if corrupted_images:
            results['issues'].append(f"Corrupted images found: {', '.join(corrupted_images[:5])}")
            if len(corrupted_images) > 5:
                results['issues'].append(f"... and {len(corrupted_images) - 5} more corrupted images")
        
        # Check annotation coverage
        image_stems = {f.stem for f in image_files}
        annotation_stems = {f.stem for f in annotation_files}
        
        missing_annotations = image_stems - annotation_stems
        if missing_annotations:
            results['issues'].append(f"Images without annotations: {len(missing_annotations)}")
        
        orphaned_annotations = annotation_stems - image_stems
        if orphaned_annotations:
            results['issues'].append(f"Annotations without images: {len(orphaned_annotations)}")
        
        # Parse annotations to get classes
        all_classes = set()
        
        for ann_file in annotation_files:
            try:
                if ann_file.suffix.lower() == '.txt':
                    # YOLO format
                    with open(ann_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_id = int(parts[0])
                                all_classes.add(class_id)
                
                elif ann_file.suffix.lower() == '.json':
                    # COCO format
                    with open(ann_file, 'r') as f:
                        data = json.load(f)
                        if 'annotations' in data:
                            for ann in data['annotations']:
                                if 'category_id' in ann:
                                    all_classes.add(ann['category_id'])
                
                elif ann_file.suffix.lower() == '.xml':
                    # Pascal VOC format
                    tree = ET.parse(ann_file)
                    root = tree.getroot()
                    for obj in root.findall('object'):
                        name = obj.find('name')
                        if name is not None:
                            all_classes.add(name.text)
            
            except Exception as e:
                results['issues'].append(f"Error parsing {ann_file.name}: {str(e)}")
        
        results['stats']['classes'] = sorted(list(all_classes))
        
        # Check for minimum requirements
        if results['stats']['total_images'] < 10:
            results['issues'].append("Very few images (recommended: 100+ per class)")
        
        if not all_classes:
            results['issues'].append("No valid class annotations found")
        
        # Set overall validity
        if results['issues']:
            results['valid'] = False
        
    except Exception as e:
        results['valid'] = False
        results['issues'].append(f"Validation error: {str(e)}")
    
    return results

def create_dataset_info(dataset_path, name, description):
    """
    Create dataset information file
    
    Args:
        dataset_path (Path): Path to dataset directory
        name (str): Dataset name
        description (str): Dataset description
        
    Returns:
        dict: Dataset information
    """
    validation_results = validate_dataset(dataset_path)
    
    # Count class distribution for YOLO format
    class_distribution = {}
    annotation_files = [f for f in dataset_path.iterdir() if f.suffix.lower() == '.txt']
    
    for ann_file in annotation_files:
        try:
            with open(ann_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        class_distribution[class_id] = class_distribution.get(class_id, 0) + 1
        except Exception:
            continue
    
    dataset_info = {
        'name': name,
        'description': description,
        'total_images': validation_results['stats']['total_images'],
        'total_annotations': validation_results['stats']['total_annotations'],
        'classes': validation_results['stats']['classes'],
        'class_distribution': class_distribution,
        'image_formats': validation_results['stats']['image_formats'],
        'annotation_formats': validation_results['stats']['annotation_formats'],
        'validation_passed': validation_results['valid'],
        'issues': validation_results['issues']
    }
    
    # Save dataset info
    info_file = dataset_path / 'dataset_info.json'
    with open(info_file, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    return dataset_info

def visualize_annotations(dataset_path, max_images=5):
    """
    Visualize sample annotations from the dataset
    
    Args:
        dataset_path (Path): Path to dataset directory
        max_images (int): Maximum number of images to visualize
        
    Returns:
        list: List of PIL Images with bounding boxes drawn
    """
    from PIL import ImageDraw
    
    visualized_images = []
    
    # Get image files
    image_files = [f for f in dataset_path.iterdir() 
                  if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
    
    for i, img_file in enumerate(image_files[:max_images]):
        try:
            # Load image
            image = Image.open(img_file)
            draw = ImageDraw.Draw(image)
            
            # Look for corresponding annotation
            ann_file = dataset_path / f"{img_file.stem}.txt"
            
            if ann_file.exists():
                with open(ann_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # YOLO format: class_id center_x center_y width height
                            class_id = int(parts[0])
                            center_x = float(parts[1])
                            center_y = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            # Convert to pixel coordinates
                            img_width, img_height = image.size
                            
                            x1 = int((center_x - width/2) * img_width)
                            y1 = int((center_y - height/2) * img_height)
                            x2 = int((center_x + width/2) * img_width)
                            y2 = int((center_y + height/2) * img_height)
                            
                            # Draw bounding box
                            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
                            draw.text((x1, y1-15), f"Class {class_id}", fill='red')
            
            visualized_images.append(image)
            
        except Exception as e:
            print(f"Error visualizing {img_file}: {e}")
            continue
    
    return visualized_images

def split_dataset(dataset_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split dataset into train/validation/test sets
    
    Args:
        dataset_path (Path): Path to dataset directory
        train_ratio (float): Training set ratio
        val_ratio (float): Validation set ratio
        test_ratio (float): Test set ratio
    """
    import random
    
    # Get all image files
    image_files = [f for f in dataset_path.iterdir() 
                  if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
    
    # Shuffle files
    random.shuffle(image_files)
    
    # Calculate split indices
    total_files = len(image_files)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)
    
    # Split files
    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]
    
    # Create split directories
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    for split_name, files in splits.items():
        split_dir = dataset_path / split_name
        split_dir.mkdir(exist_ok=True)
        
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        
        for img_file in files:
            # Copy image
            shutil.copy2(img_file, images_dir / img_file.name)
            
            # Copy corresponding annotation if exists
            ann_file = dataset_path / f"{img_file.stem}.txt"
            if ann_file.exists():
                shutil.copy2(ann_file, labels_dir / f"{img_file.stem}.txt")
    
    return {
        'train': len(train_files),
        'val': len(val_files),
        'test': len(test_files)
    }
