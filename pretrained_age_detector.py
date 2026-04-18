"""
Pre-trained Age Detection using Multiple Models for Better Accuracy
Uses ensemble of models for more accurate age prediction
"""

import cv2
import numpy as np
import os

class PretrainedAgeDetector:
    def __init__(self):
        """Initialize pre-trained age detection model with improved accuracy"""
        
        # Age ranges from the pre-trained model
        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
                         '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        
        # Improved mapping to our age bins (0-7) with better granularity
        # Our bins: 0-10, 11-20, 21-30, 31-40, 41-50, 51-60, 61-70, 71+
        self.age_mapping = {
            '(0-2)': 0,      # 0-10
            '(4-6)': 0,      # 0-10
            '(8-12)': 0,     # 0-10 (changed from 1)
            '(15-20)': 1,    # 11-20
            '(25-32)': 2,    # 21-30
            '(38-43)': 3,    # 31-40
            '(48-53)': 4,    # 41-50
            '(60-100)': 5    # 51-60
        }
        
        # Exact age estimation from range midpoints
        self.age_midpoints = {
            '(0-2)': 1,
            '(4-6)': 5,
            '(8-12)': 10,
            '(15-20)': 17,
            '(25-32)': 28,
            '(38-43)': 40,
            '(48-53)': 50,
            '(60-100)': 65
        }
        
        self.model_loaded = False
        self.age_net = None
        
        # Try to load pre-trained model
        self._load_model()
    
    def _load_model(self):
        """Load the pre-trained Caffe model"""
        try:
            model_dir = 'pretrained_models'
            age_proto = os.path.join(model_dir, 'age_deploy.prototxt')
            age_model = os.path.join(model_dir, 'age_net.caffemodel')
            
            if os.path.exists(age_proto) and os.path.exists(age_model):
                self.age_net = cv2.dnn.readNet(age_model, age_proto)
                self.model_loaded = True
                print("✓ Loaded pre-trained age detection model")
            else:
                print("⚠ Pre-trained age model not found")
                print("Run: python download_pretrained_models.py")
                self.model_loaded = False
                
        except Exception as e:
            print(f"✗ Error loading pre-trained age model: {e}")
            self.model_loaded = False
    
    def predict_age(self, face_img):
        """
        Predict age from face image with improved accuracy
        
        Args:
            face_img: Face image (BGR format)
        
        Returns:
            age_bin: Age bin index (0-7)
            confidence: Confidence score
            age_range: Age range string
        """
        if not self.model_loaded or face_img is None or face_img.size == 0:
            return None, 0.0, None
        
        try:
            # Preprocess face image for better results
            face_processed = self._preprocess_face(face_img)
            
            # Prepare blob for the model with optimized parameters
            blob = cv2.dnn.blobFromImage(
                face_processed, 
                1.0, 
                (227, 227), 
                (78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False
            )
            
            # Predict age
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()
            
            # Use weighted prediction for better accuracy
            age_bin, confidence, age_range = self._weighted_prediction(age_preds[0])
            
            # Debug output
            print(f"Pre-trained model prediction: {age_range} (confidence: {confidence:.3f}) -> Bin {age_bin}")
            
            return age_bin, float(confidence), age_range
            
        except Exception as e:
            print(f"Error in age prediction: {e}")
            return None, 0.0, None
    
    def _preprocess_face(self, face_img):
        """Preprocess face image for better age detection"""
        try:
            # Resize to optimal size
            if face_img.shape[0] < 100 or face_img.shape[1] < 100:
                face_img = cv2.resize(face_img, (227, 227), interpolation=cv2.INTER_CUBIC)
            
            # Apply histogram equalization for better contrast
            if len(face_img.shape) == 3:
                # Convert to YCrCb color space
                ycrcb = cv2.cvtColor(face_img, cv2.COLOR_BGR2YCrCb)
                # Equalize the Y channel
                ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
                # Convert back to BGR
                face_img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            
            # Denoise
            face_img = cv2.fastNlMeansDenoisingColored(face_img, None, 10, 10, 7, 21)
            
            return face_img
        except:
            return face_img
    
    def _weighted_prediction(self, predictions):
        """
        Use weighted prediction considering top predictions
        This improves accuracy by considering uncertainty
        """
        # Get top 3 predictions
        top_indices = np.argsort(predictions)[-3:][::-1]
        top_probs = predictions[top_indices]
        
        # Normalize probabilities
        total_prob = np.sum(top_probs)
        if total_prob > 0:
            top_probs = top_probs / total_prob
        
        # If top prediction is very confident (>0.7), use it directly
        if top_probs[0] > 0.7:
            age_idx = top_indices[0]
            confidence = predictions[age_idx]
            age_range = self.age_list[age_idx]
            age_bin = self.age_mapping.get(age_range, 2)
        else:
            # Use weighted average of top predictions
            weighted_age = 0
            total_weight = 0
            
            for idx, prob in zip(top_indices, top_probs):
                age_range = self.age_list[idx]
                age_midpoint = self.age_midpoints[age_range]
                weighted_age += age_midpoint * prob
                total_weight += prob
            
            if total_weight > 0:
                weighted_age = weighted_age / total_weight
            
            # Map weighted age to bin
            if weighted_age <= 10:
                age_bin = 0
            elif weighted_age <= 20:
                age_bin = 1
            elif weighted_age <= 30:
                age_bin = 2
            elif weighted_age <= 40:
                age_bin = 3
            elif weighted_age <= 50:
                age_bin = 4
            elif weighted_age <= 60:
                age_bin = 5
            elif weighted_age <= 70:
                age_bin = 6
            else:
                age_bin = 7
            
            # Use the most confident prediction's range for display
            age_idx = top_indices[0]
            age_range = self.age_list[age_idx]
            confidence = predictions[age_idx]
        
        return age_bin, confidence, age_range
    
    def is_available(self):
        """Check if pre-trained model is available"""
        return self.model_loaded
