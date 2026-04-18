"""
Face Analysis Utilities for Age and Emotion Estimation
Provides heuristic-based analysis when model is untrained
"""

import cv2
import numpy as np

# Age labels for debugging output
AGE_LABELS = [
    "0-10", "11-20", "21-30", "31-40",
    "41-50", "51-60", "61-70", "71+"
]


def analyze_face_features(face_img):
    """
    Analyze face features to estimate age using image processing heuristics
    
    Args:
        face_img: Face image (BGR format)
    
    Returns:
        estimated_age_bin: Age bin index (0-7)
        confidence: Confidence score
    """
    if face_img is None or face_img.size == 0:
        return 2, 0.5  # Default to 21-30
    
    # Convert to grayscale
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Feature 1: Skin texture analysis (wrinkles, smoothness)
    # Use Laplacian variance to detect texture complexity
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Feature 2: Face brightness (children often have brighter, smoother skin)
    mean_brightness = np.mean(gray)
    
    # Feature 3: Contrast (older faces tend to have more contrast)
    contrast = gray.std()
    
    # Feature 4: Edge density (more edges = more wrinkles/texture)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Feature 5: Face size ratio (babies/children have different proportions)
    aspect_ratio = w / h if h > 0 else 1.0
    
    # Feature 6: Skin smoothness in cheek area (center region)
    cheek_region = gray[int(h*0.3):int(h*0.7), int(w*0.2):int(w*0.8)]
    if cheek_region.size > 0:
        cheek_smoothness = cheek_region.std()
    else:
        cheek_smoothness = contrast
    
    # Scoring system (start at 0 for youngest)
    age_score = 0
    
    # TEXTURE ANALYSIS - Most important for age
    # Very smooth skin (babies/young children)
    if laplacian_var < 150:
        age_score -= 3  # Very strong indicator of baby/young child
    elif laplacian_var < 250:
        age_score -= 1  # Smooth skin, likely young
    elif laplacian_var < 400:
        age_score += 0  # Normal adult skin
    elif laplacian_var < 600:
        age_score += 2  # Textured skin, middle age
    else:
        age_score += 3  # Very textured, older age
    
    # BRIGHTNESS ANALYSIS
    # Babies and children often have brighter, more reflective skin
    if mean_brightness > 145:
        age_score -= 2  # Very bright = very young
    elif mean_brightness > 130:
        age_score -= 1  # Bright = younger
    elif mean_brightness < 100:
        age_score += 1  # Darker = potentially older
    
    # CONTRAST ANALYSIS
    # Lower contrast = smoother, younger face
    if contrast < 35:
        age_score -= 1  # Low contrast = smooth = young
    elif contrast < 50:
        age_score += 0  # Normal
    elif contrast > 65:
        age_score += 2  # High contrast = defined features = older
    else:
        age_score += 1
    
    # EDGE DENSITY (wrinkles, facial lines)
    if edge_density < 0.08:
        age_score -= 1  # Very smooth, few edges = young
    elif edge_density < 0.12:
        age_score += 0  # Normal
    elif edge_density > 0.18:
        age_score += 2  # Many edges = wrinkles = older
    else:
        age_score += 1
    
    # CHEEK SMOOTHNESS
    # Babies have very smooth cheeks
    if cheek_smoothness < 30:
        age_score -= 1  # Very smooth cheeks = young
    elif cheek_smoothness > 55:
        age_score += 1  # Textured cheeks = older
    
    # Map score to age bin (0-7)
    # Score range: approximately -9 to 9
    if age_score <= -4:
        age_bin = 0  # 0-10 (baby/child)
    elif age_score <= -2:
        age_bin = 1  # 11-20 (teen)
    elif age_score <= 0:
        age_bin = 2  # 21-30 (young adult)
    elif age_score <= 2:
        age_bin = 3  # 31-40 (adult)
    elif age_score <= 4:
        age_bin = 4  # 41-50 (middle age)
    elif age_score <= 6:
        age_bin = 5  # 51-60 (senior)
    elif age_score <= 8:
        age_bin = 6  # 61-70 (elderly)
    else:
        age_bin = 7  # 71+ (very elderly)
    
    # Calculate confidence based on feature consistency
    confidence = min(0.75, 0.5 + (abs(laplacian_var - 300) / 1000))
    
    # Debug output
    print(f"Age Analysis - Laplacian: {laplacian_var:.1f}, Brightness: {mean_brightness:.1f}, "
          f"Contrast: {contrast:.1f}, Edges: {edge_density:.3f}, Score: {age_score}, "
          f"Predicted Bin: {age_bin} ({AGE_LABELS[age_bin]})")
    
    return age_bin, confidence


def analyze_face_emotion(face_img):
    """
    Analyze face for emotion using basic image processing
    
    Args:
        face_img: Face image (BGR format)
    
    Returns:
        emotion_idx: Emotion index (0-6)
        confidence: Confidence score
    """
    if face_img is None or face_img.size == 0:
        return 6, 0.5  # Default to Neutral
    
    # Convert to grayscale
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Try to detect smile using mouth region (bottom third of face)
    h, w = gray.shape
    mouth_region = gray[int(h*0.6):, :]
    
    # Detect edges in mouth region
    edges = cv2.Canny(mouth_region, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Analyze brightness in eye region (top third)
    eye_region = gray[:int(h*0.4), :]
    eye_brightness = np.mean(eye_region)
    
    # Simple heuristic: high edge density in mouth = smile
    if edge_density > 0.12:
        return 3, 0.6  # Happy
    elif edge_density < 0.05:
        return 4, 0.5  # Sad
    else:
        return 6, 0.7  # Neutral (most common)


def estimate_age_from_face_size(face_bbox, image_shape):
    """
    Estimate age based on face size relative to image
    (Larger faces in frame often indicate closer subjects)
    
    Args:
        face_bbox: [x1, y1, x2, y2]
        image_shape: (height, width)
    
    Returns:
        age_bin: Estimated age bin (0-7)
    """
    x1, y1, x2, y2 = face_bbox
    h, w = image_shape[:2]
    
    face_area = (x2 - x1) * (y2 - y1)
    image_area = h * w
    face_ratio = face_area / image_area
    
    # Heuristic based on typical photo composition
    if face_ratio > 0.25:  # Very large face (close-up)
        return 2  # 21-30 (young adult selfies)
    elif face_ratio > 0.15:  # Large face
        return 3  # 31-40 (adult)
    elif face_ratio > 0.08:  # Medium face
        return 2  # 21-30 (young adult)
    else:  # Small face (far from camera)
        return 3  # 31-40 (adult)


def hybrid_age_estimation(face_img, face_bbox, image_shape, model_prediction=None, model_confidence=0.0):
    """
    Hybrid age estimation combining model prediction and heuristics
    
    Args:
        face_img: Face image
        face_bbox: Face bounding box
        image_shape: Original image shape
        model_prediction: Model's age bin prediction (if available)
        model_confidence: Model's confidence score
    
    Returns:
        age_bin: Final age bin estimate
        confidence: Confidence score
    """
    # Get heuristic-based estimate
    heuristic_age, heuristic_conf = analyze_face_features(face_img)
    
    # For untrained models, confidence will be around 0.125 (1/8 random guess)
    # Only trust model if confidence is significantly above random (> 0.5)
    if model_confidence > 0.5 and model_prediction is not None:
        # Model is likely trained, use its prediction
        return model_prediction, model_confidence
    
    # Model is untrained or low confidence, use heuristics
    print(f"Using heuristic age estimation (model confidence: {model_confidence:.3f})")
    return heuristic_age, heuristic_conf
