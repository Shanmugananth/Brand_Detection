#!/usr/bin/env python3
"""
Flask Backend for YOLO Object Detection Web Interface
Integrates the existing YOLO detection code for web-based image processing
"""

import os
import sys
import io
import base64
import tempfile
from collections import defaultdict
import json

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from ultralytics import YOLO

# Flask app configuration
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
MODEL_PATH = 'yolov8s.pt'  # Updated for YOLOv8s model

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global model variable (initialized to None)
model = None
labels = None

# Bounding box colors (using the Tableau 10 color scheme from original code)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

def load_model():
    """Load YOLO model at startup"""
    global model, labels, MODEL_PATH
    try:
        print(f"Current working directory: {os.getcwd()}")
        print(f"Files in current directory: {os.listdir('.')}")
        
        if not os.path.exists(MODEL_PATH):
            print(f'ERROR: Model path {MODEL_PATH} is invalid or model was not found.')
            print('Available files:', [f for f in os.listdir('.') if f.endswith('.pt')])
            # Try to find any .pt file
            pt_files = [f for f in os.listdir('.') if f.endswith('.pt')]
            if pt_files:
                MODEL_PATH = pt_files[0]
                print(f'Using model: {MODEL_PATH}')
            else:
                raise FileNotFoundError(f'No YOLO model found. Please ensure your .pt file is included in the Docker image.')
        
        print(f"Loading YOLOv8s model from: {MODEL_PATH}")
        # Load with optimizations for larger models
        model = YOLO(MODEL_PATH, task='detect')
        labels = model.names

        # Warm up the model with a small dummy inference to load weights into memory
        print("Warming up model...")
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = model(dummy_image, verbose=False)
        
        print(f'YOLOv8s model loaded successfully from {MODEL_PATH}')
        print(f'Available classes: {list(labels.values())}')
        print(f'Model size: YOLOv8s (Small)')
    except Exception as e:
        print(f'Error loading YOLOv8s model: {e}')
        print(f'Stack trace: {str(e)}')
        raise e

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image_detection(image_path, min_thresh=0.5):
    """
    Process image with YOLO detection (adapted from original code)
    Returns annotated image and detection statistics
    """
    global model, labels
    
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError("Could not read image file")
    
    # Get original dimensions
    original_height, original_width = frame.shape[:2]
    
    # Run inference on frame
    results = model(frame, verbose=False)
    
    # Extract results
    detections = results[0].boxes
    
    # Initialize statistics
    object_count = 0
    object_counts = defaultdict(int)
    confidence_scores = []
    
    # Process each detection
    if detections is not None and len(detections) > 0:
        for i in range(len(detections)):
            # Get bounding box coordinates
            xyxy_tensor = detections[i].xyxy.cpu()
            xyxy = xyxy_tensor.numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)
            
            # Get bounding box class ID and name
            classidx = int(detections[i].cls.item())
            classname = labels[classidx]
            
            # Get bounding box confidence
            conf = detections[i].conf.item()
            
            # Draw box if confidence threshold is high enough
            if conf > min_thresh:
                # Choose color based on class
                color = bbox_colors[classidx % len(bbox_colors)]
                
                # Draw bounding box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                
                # Prepare label
                label = f'{classname}: {int(conf*100)}%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_ymin = max(ymin, labelSize[1] + 10)
                
                # Draw label background
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), 
                             (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
                
                # Draw label text
                cv2.putText(frame, label, (xmin, label_ymin-7), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Update statistics
                object_count += 1
                object_counts[classname] += 1
                confidence_scores.append(conf)
    
    # Add summary text to image
    cv2.putText(frame, f'Objects Detected: {object_count}', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Calculate average confidence
    avg_confidence = round(np.mean(confidence_scores) * 100, 1) if confidence_scores else 0
    
    # Prepare detection statistics
    detection_stats = {
        'total_objects': object_count,
        'object_counts': dict(object_counts),
        'avg_confidence': avg_confidence,
        'image_dimensions': {
            'width': original_width,
            'height': original_height
        }
    }
    
    return frame, detection_stats

def encode_image_to_base64(image):
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return image_base64

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'yolo_web_interface.html')

@app.route('/detect', methods=['POST'])
def detect_objects():
    """Handle image upload and object detection"""
    global model, labels
    
    try:
        # Ensure model is loaded
        if model is None:
            load_model()
            
        # Check if file is present
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'File type not allowed'}), 400
        
        # Get confidence threshold from request
        min_thresh = float(request.form.get('threshold', 0.5))
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(temp_path)
        
        try:
            # Process image with YOLO detection
            annotated_image, detection_stats = process_image_detection(temp_path, min_thresh)
            
            # Convert annotated image to base64
            image_base64 = encode_image_to_base64(annotated_image)
            
            # Clean up temporary file
            os.remove(temp_path)
            
            # Return results
            return jsonify({
                'success': True,
                'image': image_base64,
                'detections': detection_stats,
                'message': f'Successfully detected {detection_stats["total_objects"]} objects'
            })
            
        except Exception as e:
            # Clean up temporary file in case of error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e
            
    except Exception as e:
        return jsonify({
            'success': False, 
            'error': f'Detection failed: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global model, labels
    
    try:
        if model is None:
            load_model()
        
        return jsonify({
            'status': 'healthy',
            'model_loaded': model is not None,
            'available_classes': list(labels.values()) if labels else []
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'model_loaded': False
        }), 500

@app.errorhandler(413)
def file_too_large(error):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB.'
    }), 413

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error occurred.'
    }), 500

# Initialize model with error handling
print("=== YOLO Object Detection Service Starting ===")
try:
    model_loaded = load_model()
    if model_loaded:
        print("✓ Model loaded successfully at startup")
    else:
        print("⚠ Model loading skipped - will load on first request")
except Exception as e:
    print(f"⚠ Model loading failed at startup: {e}")
    print("Service will continue with lazy loading...")
    model = None
    labels = None

if __name__ == '__main__':
    print("Starting YOLO Object Detection Web Server...")
    print("=" * 50)
    
    # For local development, try to load model
    if model is None:
        print("Loading model for local development...")
        try:
            success = load_model()
            if success:
                print("✓ Model loaded successfully!")
            else:
                print("⚠ Model not loaded - check model file path")
        except Exception as e:
            print(f"⚠ Model loading failed: {e}")
            print("Service will continue with lazy loading...")
    
    print("=" * 50)
    print("Server starting on http://localhost:5000")
    print("Open http://localhost:5000 in your browser")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Get port from environment variable (for GCP deployment)
    port = int(os.environ.get('PORT', 5000))
    
    # Run Flask app with debug mode for local development
    app.run(debug=True, host='0.0.0.0', port=port)
else:
    # For production deployment with gunicorn
    print("Production mode - Flask app initialized")
