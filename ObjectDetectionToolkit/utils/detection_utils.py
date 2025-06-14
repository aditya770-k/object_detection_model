import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os

def load_model(model_path):
    """
    Load YOLO model for inference
    
    Args:
        model_path (str): Path to model file
        
    Returns:
        Model object or None if ultralytics not available
    """
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        return model
    except ImportError:
        raise Exception("Ultralytics not installed. Please install with: pip install ultralytics")
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

def detect_objects(model, image, config):
    """
    Detect objects in image using YOLO model
    
    Args:
        model: Loaded YOLO model
        image (np.array): Input image
        config (dict): Detection configuration
        
    Returns:
        list: Detection results
    """
    try:
        # Run inference
        results = model(
            image,
            conf=config['confidence'],
            iou=config['iou'],
            max_det=config['max_det'],
            imgsz=config['imgsz']
        )
        
        return results
        
    except Exception as e:
        raise Exception(f"Error during detection: {str(e)}")

def draw_detections(image, results, class_names=None):
    """
    Draw bounding boxes and labels on image
    
    Args:
        image (np.array): Input image
        results: Detection results from YOLO
        class_names (list): Optional class names
        
    Returns:
        np.array: Image with drawn detections
    """
    try:
        # Convert to PIL for easier drawing
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image.copy()
        
        draw = ImageDraw.Draw(pil_image)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Colors for different classes
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (128, 0, 0),    # Maroon
            (0, 128, 0),    # Olive
            (0, 0, 128),    # Navy
            (128, 128, 0),  # Olive
        ]
        
        # Draw detections
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    # Get box coordinates
                    if hasattr(box, 'xyxy'):
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    else:
                        continue
                    
                    # Get class and confidence
                    if hasattr(box, 'cls'):
                        class_id = int(box.cls.cpu().numpy()[0])
                    else:
                        class_id = 0
                    
                    if hasattr(box, 'conf'):
                        confidence = float(box.conf.cpu().numpy()[0])
                    else:
                        confidence = 1.0
                    
                    # Select color
                    color = colors[class_id % len(colors)]
                    
                    # Draw bounding box
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                    
                    # Prepare label
                    if class_names and class_id < len(class_names):
                        label = f"{class_names[class_id]}: {confidence:.2f}"
                    else:
                        label = f"Class {class_id}: {confidence:.2f}"
                    
                    # Draw label background
                    text_bbox = draw.textbbox((x1, y1), label, font=font)
                    draw.rectangle(text_bbox, fill=color)
                    
                    # Draw label text
                    draw.text((x1, y1), label, fill=(255, 255, 255), font=font)
        
        # Convert back to numpy array
        return np.array(pil_image)
        
    except Exception as e:
        # Return original image if drawing fails
        print(f"Error drawing detections: {e}")
        return image

def process_video(model, video_path, output_path, config):
    """
    Process video for object detection
    
    Args:
        model: Loaded YOLO model
        video_path (str): Input video path
        output_path (str): Output video path
        config (dict): Detection configuration
        
    Returns:
        dict: Processing results
    """
    try:
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        detection_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            results = detect_objects(model, frame, config)
            
            # Draw detections
            frame_with_detections = draw_detections(frame, results)
            
            # Count detections
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    detection_count += len(result.boxes)
            
            # Write frame
            out.write(frame_with_detections)
            frame_count += 1
        
        cap.release()
        out.release()
        
        return {
            'success': True,
            'total_frames': frame_count,
            'total_detections': detection_count,
            'output_path': output_path
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def batch_detect(model, image_paths, config, output_dir=None):
    """
    Perform batch detection on multiple images
    
    Args:
        model: Loaded YOLO model
        image_paths (list): List of image file paths
        config (dict): Detection configuration
        output_dir (str): Optional output directory for results
        
    Returns:
        list: Detection results for all images
    """
    results = []
    
    for img_path in image_paths:
        try:
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                results.append({
                    'image_path': img_path,
                    'success': False,
                    'error': 'Could not load image'
                })
                continue
            
            # Run detection
            detections = detect_objects(model, image, config)
            
            # Process results
            processed_detections = []
            for result in detections:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        detection = {
                            'bbox': box.xyxy[0].cpu().numpy().tolist() if hasattr(box, 'xyxy') else [],
                            'confidence': float(box.conf.cpu().numpy()[0]) if hasattr(box, 'conf') else 0.0,
                            'class_id': int(box.cls.cpu().numpy()[0]) if hasattr(box, 'cls') else 0
                        }
                        processed_detections.append(detection)
            
            results.append({
                'image_path': img_path,
                'success': True,
                'detections': processed_detections,
                'detection_count': len(processed_detections)
            })
            
            # Save annotated image if output directory provided
            if output_dir:
                annotated_image = draw_detections(image, detections)
                output_path = os.path.join(output_dir, f"detected_{os.path.basename(img_path)}")
                cv2.imwrite(output_path, annotated_image)
                results[-1]['output_path'] = output_path
            
        except Exception as e:
            results.append({
                'image_path': img_path,
                'success': False,
                'error': str(e)
            })
    
    return results

def calculate_detection_metrics(detections, ground_truth, iou_threshold=0.5):
    """
    Calculate detection metrics (precision, recall, mAP)
    
    Args:
        detections (list): List of detection results
        ground_truth (list): List of ground truth annotations
        iou_threshold (float): IoU threshold for matching
        
    Returns:
        dict: Calculated metrics
    """
    try:
        # Simplified metrics calculation
        # In a real implementation, this would be more comprehensive
        
        total_detections = len(detections)
        total_ground_truth = len(ground_truth)
        
        # Match detections to ground truth (simplified)
        true_positives = 0
        
        for detection in detections:
            for gt in ground_truth:
                iou = calculate_iou(detection['bbox'], gt['bbox'])
                if iou >= iou_threshold and detection['class_id'] == gt['class_id']:
                    true_positives += 1
                    break
        
        false_positives = total_detections - true_positives
        false_negatives = total_ground_truth - true_positives
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
        
    except Exception as e:
        return {
            'error': str(e)
        }

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes
    
    Args:
        box1 (list): [x1, y1, x2, y2]
        box2 (list): [x1, y1, x2, y2]
        
    Returns:
        float: IoU value
    """
    try:
        # Calculate intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        if union <= 0:
            return 0.0
        
        return intersection / union
        
    except Exception:
        return 0.0
