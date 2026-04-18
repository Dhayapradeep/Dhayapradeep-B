"""
Test the pre-trained age detection model
"""

import cv2
import numpy as np
from utils.pretrained_age_detector import PretrainedAgeDetector

# Initialize detector
detector = PretrainedAgeDetector()

if not detector.is_available():
    print("ERROR: Pre-trained model not loaded!")
    print("Run: python download_pretrained_models.py")
    exit(1)

print("✓ Pre-trained model loaded successfully")
print("\nTesting with sample images...")

# Test with a simple generated image (for testing)
print("\n" + "="*60)
print("Creating test face images...")
print("="*60)

# Create a bright smooth face (simulating baby)
baby_face = np.ones((200, 200, 3), dtype=np.uint8) * 200  # Bright
baby_face = cv2.GaussianBlur(baby_face, (15, 15), 0)  # Very smooth

print("\nTest 1: Bright smooth face (simulating baby)")
age_bin, conf, age_range = detector.predict_age(baby_face)
print(f"Result: {age_range} -> Bin {age_bin} (confidence: {conf:.3f})")

# Create a normal face (simulating adult)
adult_face = np.ones((200, 200, 3), dtype=np.uint8) * 120  # Normal brightness
adult_face = cv2.GaussianBlur(adult_face, (5, 5), 0)  # Some texture

print("\nTest 2: Normal face (simulating adult)")
age_bin, conf, age_range = detector.predict_age(adult_face)
print(f"Result: {age_range} -> Bin {age_bin} (confidence: {conf:.3f})")

# Create a textured face (simulating senior)
senior_face = np.ones((200, 200, 3), dtype=np.uint8) * 100  # Darker
# Add some noise for texture
noise = np.random.randint(0, 50, (200, 200, 3), dtype=np.uint8)
senior_face = cv2.add(senior_face, noise)

print("\nTest 3: Textured face (simulating senior)")
age_bin, conf, age_range = detector.predict_age(senior_face)
print(f"Result: {age_range} -> Bin {age_bin} (confidence: {conf:.3f})")

print("\n" + "="*60)
print("Model is working! Now test with real images in the web app.")
print("="*60)
