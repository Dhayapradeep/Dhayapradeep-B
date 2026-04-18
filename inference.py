import torch
import cv2
import numpy as np
from models.multitask_model import MultiTaskFaceModel
from models.roi_heads import AGE_LABELS, EMOTION_LABELS, decode_predictions
import torch.nn.functional as F
import os

class FacialAnalysisInference:
    def __init__(self, model_path, device):
        self.device = device
        self.model = MultiTaskFaceModel().to(device)
        
        # Load trained weights if available
        try:
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model from {model_path}")
            else:
                print(f"Model file {model_path} not found. Using randomly initialized weights.")
                print("Note: You need to train the model first for accurate results.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using randomly initialized weights.")
        
        self.model.eval()
        
        # Try to load OpenCV DNN face detector, fallback to Haar cascades
        try:
            # You would need to download these files from OpenCV repository
            self.face_net = cv2.dnn.readNetFromTensorflow('opencv_face_detector_uint8.pb', 
                                                         'opencv_face_detector.pbtxt')
            self.use_dnn = True
            print("Using OpenCV DNN face detector")
        except:
            # Fallback to Haar cascades
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.use_dnn = False
            print("Using Haar cascade face detector")
    
    def detect_faces(self, image):
        """Detect faces in the image"""
        if self.use_dnn:
            return self._detect_faces_dnn(image)
        else:
            return self._detect_faces_haar(image)
    
    def _detect_faces_dnn(self, image):
        """Detect faces using OpenCV DNN"""
        h, w = image.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Confidence threshold
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                faces.append([x1, y1, x2, y2, confidence])
        
        return faces
    
    def _detect_faces_haar(self, image):
        """Detect faces using Haar cascades"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Improved parameters for better detection
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        face_list = []
        for (x, y, w, h) in faces:
            face_list.append([x, y, x+w, y+h, 0.9])  # Dummy confidence
        
        print(f"Detected {len(face_list)} faces using Haar cascade")
        return face_list
    
    def preprocess_face(self, face_img):
        """Preprocess face image for the model"""
        # Resize to model input size (assuming 224x224)
        face_resized = cv2.resize(face_img, (224, 224))
        
        # Normalize
        face_normalized = face_resized.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        face_tensor = torch.from_numpy(face_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return face_tensor.to(self.device)
    
    def analyze_faces(self, image):
        """Analyze faces in the image for age and emotion"""
        print(f"Input image shape: {image.shape}")
        
        # Detect faces
        faces = self.detect_faces(image)
        print(f"Total faces detected: {len(faces)}")
        
        if not faces:
            return {
                'num_faces': 0,
                'faces': [],
                'message': 'No faces detected in the image'
            }
        
        results = []
        
        with torch.no_grad():
            for i, (x1, y1, x2, y2, conf) in enumerate(faces):
                print(f"Processing face {i+1}: bbox=({x1}, {y1}, {x2}, {y2})")
                
                # Extract face region
                face_img = image[y1:y2, x1:x2]
                
                if face_img.size == 0:
                    print(f"Face {i+1} has zero size, skipping")
                    continue
                
                print(f"Face {i+1} extracted, shape: {face_img.shape}")
                
                # Preprocess face
                face_tensor = self.preprocess_face(face_img)
                
                # Create dummy ROI (since we're doing inference on cropped faces)
                rois = torch.tensor([[0, 0, 0, 224, 224]], dtype=torch.float32).to(self.device)
                
                # Forward pass
                try:
                    outputs = self.model(face_tensor, rois)
                    
                    # Use decode_predictions function
                    age_label, emotion_label = decode_predictions(outputs['age'], outputs['emotion'])
                    
                    # Get confidence scores
                    age_probs = F.softmax(outputs['age'], dim=1)
                    emotion_probs = F.softmax(outputs['emotion'], dim=1)
                    
                    age_confidence = torch.max(age_probs).item()
                    emotion_confidence = torch.max(emotion_probs).item()
                    
                    face_result = {
                        'face_id': i + 1,
                        'bbox': [x1, y1, x2, y2],
                        'detection_confidence': float(conf),
                        'age': {
                            'range': age_label,
                            'confidence': float(age_confidence)
                        },
                        'emotion': {
                            'label': emotion_label,
                            'confidence': float(emotion_confidence)
                        }
                    }
                    
                    results.append(face_result)
                    print(f"Successfully processed face {i+1}: Age {age_label}, Emotion {emotion_label}")
                    
                except Exception as e:
                    print(f"Error processing face {i+1}: {str(e)}")
                    # Add a fallback result for debugging
                    face_result = {
                        'face_id': i + 1,
                        'bbox': [x1, y1, x2, y2],
                        'detection_confidence': float(conf),
                        'age': {
                            'range': 'Unknown',
                            'confidence': 0.0
                        },
                        'emotion': {
                            'label': 'Unknown',
                            'confidence': 0.0
                        },
                        'error': str(e)
                    }
                    results.append(face_result)
                    continue
        
        return {
            'num_faces': len(results),
            'faces': results,
            'message': f'Successfully analyzed {len(results)} face(s)'
        }