# =============================================================
# app.py - Advanced Web App for Multi-Task Facial Analysis
# Features:
# 1. Bounding boxes on UI
# 2. Correct face count
# 3. Live webcam support
# 4. RetinaFace-style decoding + NMS (simplified)
# =============================================================

import os
import cv2
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify, Response
from torchvision.ops import nms
from torchvision import transforms
from models.multitask_model import MultiTaskFaceModel
from models.roi_heads import AGE_LABELS, EMOTION_LABELS, decode_predictions
from utils.face_analyzer import hybrid_age_estimation, analyze_face_emotion
from utils.pretrained_age_detector import PretrainedAgeDetector

# -------------------------------------------------------------
# App Config
# -------------------------------------------------------------
app = Flask(__name__)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize pre-trained age detector
pretrained_age_detector = PretrainedAgeDetector()

# -------------------------------------------------------------
# Load Model
# -------------------------------------------------------------
model = MultiTaskFaceModel().to(DEVICE)
MODEL_TRAINED = False  # Flag to track if model is trained

try:
    model.load_state_dict(torch.load('model.pth', map_location=DEVICE))
    print("Loaded trained model from model.pth")
    MODEL_TRAINED = True
except FileNotFoundError:
    print("Model file 'model.pth' not found. Using randomly initialized weights.")
    print("Note: You need to train the model first for accurate results.")
    print("USING HEURISTIC-ONLY MODE FOR AGE/EMOTION DETECTION")
    MODEL_TRAINED = False

model.eval()

# -------------------------------------------------------------
# Image Transform
# -------------------------------------------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -------------------------------------------------------------
# Face Detection using OpenCV (Enhanced for Better Age Detection)
# -------------------------------------------------------------
def detect_faces_opencv(image):
    """Enhanced face detection with better quality face crops for age prediction"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization for better detection
    gray = cv2.equalizeHist(gray)
    
    # Use frontal face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces with optimized parameters for accuracy
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.05,     # Smaller steps for better detection
        minNeighbors=6,       # Balanced for accuracy
        minSize=(80, 80),     # Larger minimum for better quality
        maxSize=(500, 500),   # Allow larger faces
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    face_boxes = []
    h, w = image.shape[:2]
    
    for (x, y, fw, fh) in faces:
        # Additional filtering: check aspect ratio (faces are roughly square)
        aspect_ratio = fw / fh
        if 0.75 <= aspect_ratio <= 1.25:  # Face should be roughly square
            # Check if face is not too close to edges
            if x > 5 and y > 5 and (x + fw) < (w - 5) and (y + fh) < (h - 5):
                # Expand bounding box slightly for better context (10% padding)
                padding = int(fw * 0.1)
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(w, x + fw + padding)
                y2 = min(h, y + fh + padding)
                
                face_boxes.append([x1, y1, x2, y2])
    
    return face_boxes

# -------------------------------------------------------------
# RetinaFace-style Decode + NMS (Simplified)
# -------------------------------------------------------------
def decode_boxes(cls_logits, bbox_regs, score_thresh=0.6, iou_thresh=0.4):
    """Try to decode boxes from model output, fallback to OpenCV if fails"""
    try:
        boxes = []
        scores = []
        
        # NOTE: Simplified for demo – assumes single feature level
        cls_map = cls_logits[0].sigmoid()
        box_map = bbox_regs[0]
        
        _, _, H, W = cls_map.shape
        
        for y in range(H):
            for x in range(W):
                score = cls_map[0, 1, y, x]
                if score > score_thresh:
                    dx, dy, dw, dh = box_map[0, :, y, x]
                    x1 = x * 16 + dx
                    y1 = y * 16 + dy
                    x2 = x1 + dw * 16
                    y2 = y1 + dh * 16
                    boxes.append([x1, y1, x2, y2])
                    scores.append(score)
        
        if len(boxes) == 0:
            return []
        
        boxes = torch.tensor(boxes)
        scores = torch.tensor(scores)
        keep = nms(boxes, scores, iou_thresh)
        return boxes[keep].cpu().numpy()
    
    except Exception as e:
        print(f"Error in decode_boxes: {str(e)}")
        return []

# -------------------------------------------------------------
# Routes
# -------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        file = request.files['image']
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        input_tensor = transform(rgb).unsqueeze(0).to(DEVICE)
        
        # Try model-based detection first
        boxes = []
        try:
            with torch.no_grad():
                outputs = model(input_tensor)
            
            # Use correct output keys from the model
            boxes = decode_boxes(outputs['detection_cls'], outputs['detection_bbox'])
        except Exception as e:
            print(f"Model detection failed: {str(e)}")
        
        # Fallback to OpenCV if model detection fails or returns no faces
        if len(boxes) == 0:
            print("Using OpenCV fallback for face detection")
            boxes = detect_faces_opencv(image)
        
        faces = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            
            # Ensure coordinates are within image bounds
            h, w = image.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w-1))
            y2 = max(0, min(y2, h-1))
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue
            
            try:
                # Extract face ROI for analysis
                face_roi = rgb[y1:y2, x1:x2]
                face_roi_bgr = cv2.cvtColor(face_roi, cv2.COLOR_RGB2BGR)
                
                print(f"\n=== Processing Face {i+1} ===")
                print(f"Face ROI shape: {face_roi_bgr.shape}")
                print(f"Pretrained detector available: {pretrained_age_detector.is_available()}")
                
                # Try pre-trained model first (most accurate)
                if pretrained_age_detector.is_available():
                    print("Using pre-trained age detector...")
                    age_bin, confidence, age_range = pretrained_age_detector.predict_age(face_roi_bgr)
                    
                    if age_bin is not None:
                        age_label = AGE_LABELS[age_bin]
                        print(f"✓ Pre-trained model: {age_range} -> {age_label} (confidence: {confidence:.2f})")
                    else:
                        # Fallback to heuristics
                        print("Pre-trained model returned None, using heuristics...")
                        age_bin, _ = hybrid_age_estimation(face_roi_bgr, [x1, y1, x2, y2], image.shape)
                        age_label = AGE_LABELS[age_bin]
                        print(f"Heuristic result: {age_label}")
                    
                    # Emotion from heuristics (pre-trained model doesn't have emotion)
                    emotion_idx, _ = analyze_face_emotion(face_roi_bgr)
                    emotion = EMOTION_LABELS[emotion_idx]
                    
                # If model is not trained and no pre-trained model, use ONLY heuristics
                elif not MODEL_TRAINED:
                    print("Using heuristic-only mode...")
                    print(f"Processing face {i+1} with heuristic-only mode")
                    
                    # Use heuristic-only analysis
                    age_bin, age_conf = hybrid_age_estimation(
                        face_roi_bgr,
                        [x1, y1, x2, y2],
                        image.shape
                    )
                    age_label = AGE_LABELS[age_bin]
                    
                    emotion_idx, emotion_conf = analyze_face_emotion(face_roi_bgr)
                    emotion = EMOTION_LABELS[emotion_idx]
                    
                else:
                    # Model is trained, use it
                    roi = torch.tensor([[0, x1, y1, x2, y2]], dtype=torch.float).to(DEVICE)
                    
                    with torch.no_grad():
                        attr = model(input_tensor, roi)
                    
                    # Handle age and emotion prediction with hybrid approach
                    if 'age' in attr and attr['age'] is not None:
                        age_logits = attr['age']
                        emotion_logits = attr['emotion']
                        
                        # Get model predictions
                        age_probs = torch.softmax(age_logits, dim=1)
                        model_age_bin = torch.argmax(age_probs, dim=1).item()
                        model_confidence = torch.max(age_probs).item()
                        
                        # Use hybrid estimation (combines model + heuristics)
                        age_bin, confidence = hybrid_age_estimation(
                            face_roi_bgr,
                            [x1, y1, x2, y2],
                            image.shape,
                            model_age_bin,
                            model_confidence
                        )
                        
                        age_label = AGE_LABELS[age_bin]
                        
                        # Emotion prediction
                        emotion_idx = torch.argmax(emotion_logits, dim=1).item()
                        emotion = EMOTION_LABELS[emotion_idx]
                        
                    else:
                        # Fallback to heuristic-only analysis
                        age_bin, _ = hybrid_age_estimation(
                            face_roi_bgr,
                            [x1, y1, x2, y2],
                            image.shape
                        )
                        age_label = AGE_LABELS[age_bin]
                        
                        emotion_idx, _ = analyze_face_emotion(face_roi_bgr)
                        emotion = EMOTION_LABELS[emotion_idx]
                
                faces.append({
                    'box': [x1, y1, x2, y2],
                    'age': age_label,
                    'emotion': emotion
                })
                
            except Exception as e:
                print(f"Error processing face {i+1}: {str(e)}")
                # Add face with default values
                faces.append({
                    'box': [x1, y1, x2, y2],
                    'age': "21-30",
                    'emotion': "Neutral"
                })
        
        return jsonify({
            'face_count': len(faces),
            'faces': faces
        })
        
    except Exception as e:
        print(f"Error in analyze: {str(e)}")
        return jsonify({'error': str(e)}), 500

# -------------------------------------------------------------
# Live Webcam Stream (Simplified Windows Method)
# -------------------------------------------------------------
def gen_frames():
    import time
    print("Initializing camera...")
    
    cap = None
    
    # Simple, direct camera access
    try:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Camera not opened, trying with DSHOW...")
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if cap.isOpened():
            print("Camera opened successfully")
            
            # Set basic properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Give camera time to warm up
            time.sleep(1)
            
            # Discard first few frames
            for _ in range(5):
                cap.read()
            
            frame_count = 0
            
            while True:
                try:
                    ret, frame = cap.read()
                    
                    if not ret or frame is None:
                        print("Failed to read frame")
                        time.sleep(0.1)
                        continue
                    
                    # Add live indicator
                    cv2.putText(frame, 'LIVE CAMERA', (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Face detection and analysis
                    try:
                        boxes = detect_faces_opencv(frame)
                        
                        for i, box in enumerate(boxes):
                            x1, y1, x2, y2 = map(int, box)
                            
                            # Ensure coordinates are within bounds
                            h, w = frame.shape[:2]
                            x1 = max(0, min(x1, w-1))
                            y1 = max(0, min(y1, h-1))
                            x2 = max(0, min(x2, w-1))
                            y2 = max(0, min(y2, h-1))
                            
                            if x2 <= x1 or y2 <= y1:
                                continue
                            
                            # Draw face box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Predict age and emotion
                            try:
                                # Extract face ROI
                                face_roi = frame[y1:y2, x1:x2]
                                
                                if face_roi.size > 0:
                                    # Try pre-trained model first (most accurate)
                                    if pretrained_age_detector.is_available():
                                        age_bin, confidence, age_range = pretrained_age_detector.predict_age(face_roi)
                                        
                                        if age_bin is not None:
                                            age_label = AGE_LABELS[age_bin]
                                        else:
                                            # Fallback to heuristics
                                            age_bin, _ = hybrid_age_estimation(face_roi, [x1, y1, x2, y2], frame.shape)
                                            age_label = AGE_LABELS[age_bin]
                                        
                                        # Emotion from heuristics
                                        emotion_idx, _ = analyze_face_emotion(face_roi)
                                        emotion = EMOTION_LABELS[emotion_idx]
                                        
                                    # If model is not trained and no pre-trained model, use ONLY heuristics
                                    elif not MODEL_TRAINED:
                                        # Use heuristic-only analysis
                                        age_bin, _ = hybrid_age_estimation(
                                            face_roi,
                                            [x1, y1, x2, y2],
                                            frame.shape
                                        )
                                        age_label = AGE_LABELS[age_bin]
                                        
                                        emotion_idx, _ = analyze_face_emotion(face_roi)
                                        emotion = EMOTION_LABELS[emotion_idx]
                                    else:
                                        # Model is trained, use it
                                        # Convert to RGB and create tensor
                                        rgb_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                                        face_tensor = transform(rgb_roi).unsqueeze(0).to(DEVICE)
                                        
                                        # Create ROI tensor for model
                                        roi_tensor = torch.tensor([[0, x1, y1, x2, y2]], dtype=torch.float).to(DEVICE)
                                        
                                        # Get predictions
                                        with torch.no_grad():
                                            # Get full frame tensor
                                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                            frame_tensor = transform(rgb_frame).unsqueeze(0).to(DEVICE)
                                            
                                            # Get attributes
                                            attr = model(frame_tensor, roi_tensor)
                                        
                                        # Process age and emotion with hybrid approach
                                        if 'age' in attr and attr['age'] is not None:
                                            age_logits = attr['age']
                                            emotion_logits = attr['emotion']
                                            
                                            # Get model predictions
                                            age_probs = torch.softmax(age_logits, dim=1)
                                            model_age_bin = torch.argmax(age_probs, dim=1).item()
                                            model_confidence = torch.max(age_probs).item()
                                            
                                            # Use hybrid estimation
                                            age_bin, confidence = hybrid_age_estimation(
                                                face_roi,
                                                [x1, y1, x2, y2],
                                                frame.shape,
                                                model_age_bin,
                                                model_confidence
                                            )
                                            
                                            age_label = AGE_LABELS[age_bin]
                                            
                                            # Emotion prediction
                                            emotion_idx = torch.argmax(emotion_logits, dim=1).item()
                                            emotion = EMOTION_LABELS[emotion_idx]
                                        else:
                                            # Fallback to heuristic-only
                                            age_bin, _ = hybrid_age_estimation(
                                                face_roi,
                                                [x1, y1, x2, y2],
                                                frame.shape
                                            )
                                            age_label = AGE_LABELS[age_bin]
                                            
                                            emotion_idx, _ = analyze_face_emotion(face_roi)
                                            emotion = EMOTION_LABELS[emotion_idx]
                                    
                                    # Display age and emotion on frame
                                    label = f'{age_label}, {emotion}'
                                    
                                    # Draw label background
                                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                    cv2.rectangle(frame, (x1, y1-30), (x1 + label_size[0], y1), (0, 255, 0), -1)
                                    
                                    # Draw label text
                                    cv2.putText(frame, label, (x1, y1-10), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                                else:
                                    # Just show face number if ROI is invalid
                                    cv2.putText(frame, f'Face {i+1}', (x1, y1-10), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                    
                            except Exception as e:
                                print(f"Error analyzing face {i+1}: {e}")
                                # Show basic label on error
                                cv2.putText(frame, f'Face {i+1}', (x1, y1-10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    except Exception as e:
                        print(f"Face detection error: {e}")
                    
                    # Frame counter
                    cv2.putText(frame, f'Frame: {frame_count}', (10, frame.shape[0]-20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Encode
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    
                    frame_count += 1
                    
                except GeneratorExit:
                    print("Client disconnected")
                    break
                except Exception as e:
                    print(f"Frame error: {e}")
                    time.sleep(0.1)
                    continue
            
        else:
            print("Camera failed to open")
            raise Exception("Camera not accessible")
            
    except Exception as e:
        print(f"Camera error: {e}")
    finally:
        if cap is not None:
            try:
                cap.release()
                print("Camera released")
            except:
                pass
    
    # Return error frame if camera failed
    print("Returning error frame")
    error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(error_frame, 'CAMERA NOT ACCESSIBLE', (100, 200), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.putText(error_frame, 'Please check:', (200, 260), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(error_frame, '1. Camera permissions in Windows Settings', (80, 300), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(error_frame, '2. No other apps using the camera', (80, 330), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(error_frame, '3. Camera is properly connected', (80, 360), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    _, buffer = cv2.imencode('.jpg', error_frame)
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/webcam')
def webcam():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# -------------------------------------------------------------
# Run
# -------------------------------------------------------------
if __name__ == '__main__':
    print(f"Starting Multi-Task Facial Analysis Web Application on {DEVICE}...")
    app.run(debug=False, host='127.0.0.1', port=5002)