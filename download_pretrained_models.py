"""
Download pre-trained age and gender detection models
These are OpenCV DNN models trained on real datasets
"""

import urllib.request
import os

def download_file(url, filename):
    """Download a file from URL"""
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"✓ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"✗ Error downloading {filename}: {e}")
        return False

def download_age_gender_models():
    """Download pre-trained age and gender detection models"""
    
    print("="*60)
    print("DOWNLOADING PRE-TRAINED AGE DETECTION MODELS")
    print("="*60)
    print("\nThese models are trained on real face datasets and will")
    print("provide accurate age predictions without training.\n")
    
    # Create models directory
    os.makedirs('pretrained_models', exist_ok=True)
    
    # Model URLs from OpenCV's GitHub repository
    models = {
        'age_net.caffemodel': 'https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel',
        'age_deploy.prototxt': 'https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/age_deploy.prototxt',
        'gender_net.caffemodel': 'https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel',
        'gender_deploy.prototxt': 'https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/gender_deploy.prototxt'
    }
    
    # Alternative URLs (backup)
    alt_models = {
        'age_net.caffemodel': 'https://www.dropbox.com/s/xfb20y596869vbb/age_net.caffemodel?dl=1',
        'age_deploy.prototxt': 'https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt',
        'gender_net.caffemodel': 'https://www.dropbox.com/s/iyv483wz7ztr9gh/gender_net.caffemodel?dl=1',
        'gender_deploy.prototxt': 'https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt'
    }
    
    success_count = 0
    
    for filename, url in models.items():
        filepath = os.path.join('pretrained_models', filename)
        
        # Skip if already exists
        if os.path.exists(filepath):
            print(f"✓ {filename} already exists")
            success_count += 1
            continue
        
        # Try primary URL
        if download_file(url, filepath):
            success_count += 1
        else:
            # Try alternative URL
            print(f"Trying alternative URL for {filename}...")
            if download_file(alt_models[filename], filepath):
                success_count += 1
    
    print("\n" + "="*60)
    if success_count == 4:
        print("✓ ALL MODELS DOWNLOADED SUCCESSFULLY!")
        print("\nYou can now run the application with accurate age detection.")
        print("Run: python app.py")
    else:
        print(f"⚠ Downloaded {success_count}/4 models")
        print("\nIf download failed, you can manually download from:")
        print("https://github.com/spmallick/learnopencv/tree/master/AgeGender")
        print("\nPlace the files in the 'pretrained_models' folder")
    print("="*60)

if __name__ == "__main__":
    download_age_gender_models()
