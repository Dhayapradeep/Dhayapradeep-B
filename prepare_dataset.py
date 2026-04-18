"""
Dataset Preparation Script
This script helps you prepare data for training the face model
"""

import os
import urllib.request
import zipfile
import shutil

def create_data_structure():
    """Create the required data directory structure"""
    print("Creating data directory structure...")
    
    os.makedirs('data/images', exist_ok=True)
    
    # Create a sample labels.txt file
    with open('data/labels.txt', 'w') as f:
        f.write("# Format: filename,age,emotion\n")
        f.write("# Emotion codes: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral\n")
        f.write("# Example:\n")
        f.write("# face1.jpg,25,3\n")
        f.write("# face2.jpg,35,6\n")
    
    print("✓ Created data/images/ directory")
    print("✓ Created data/labels.txt template")
    print("\nNext steps:")
    print("1. Add your face images to data/images/")
    print("2. Update data/labels.txt with your image labels")
    print("3. Run: python train.py")

def download_sample_dataset():
    """
    Instructions for downloading popular face datasets
    """
    print("\n" + "="*60)
    print("RECOMMENDED DATASETS FOR TRAINING")
    print("="*60)
    
    print("\n1. UTKFace Dataset (Age, Gender, Ethnicity)")
    print("   - Size: ~23,000 images")
    print("   - Download: https://susanqq.github.io/UTKFace/")
    print("   - Format: [age]_[gender]_[race]_[date&time].jpg")
    
    print("\n2. FER2013 Dataset (Emotion Recognition)")
    print("   - Size: ~35,000 images")
    print("   - Download: https://www.kaggle.com/datasets/msambare/fer2013")
    print("   - 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral")
    
    print("\n3. IMDB-WIKI Dataset (Age and Gender)")
    print("   - Size: 500,000+ images")
    print("   - Download: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/")
    print("   - Contains celebrity faces with age labels")
    
    print("\n4. AffectNet Dataset (Emotion and Valence)")
    print("   - Size: 1,000,000+ images")
    print("   - Download: http://mohammadmahoor.com/affectnet/")
    print("   - Requires registration")
    
    print("\n" + "="*60)
    print("QUICK START WITH SAMPLE DATA")
    print("="*60)
    print("\nIf you don't have a dataset yet:")
    print("1. Collect 100-200 face images")
    print("2. Manually label them with age and emotion")
    print("3. Place in data/images/")
    print("4. Create labels.txt with format: filename,age,emotion")
    print("\nExample labels.txt:")
    print("person1.jpg,25,3")
    print("person2.jpg,35,6")
    print("person3.jpg,18,3")

def convert_utkface_dataset(utkface_dir):
    """
    Convert UTKFace dataset to our format
    UTKFace format: [age]_[gender]_[race]_[date&time].jpg
    """
    print(f"\nConverting UTKFace dataset from {utkface_dir}...")
    
    if not os.path.exists(utkface_dir):
        print(f"Error: Directory {utkface_dir} not found!")
        return
    
    os.makedirs('data/images', exist_ok=True)
    
    labels = []
    count = 0
    
    for filename in os.listdir(utkface_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            try:
                # Parse filename: age_gender_race_date.jpg
                parts = filename.split('_')
                age = int(parts[0])
                
                # Default emotion to neutral (6) since UTKFace doesn't have emotion labels
                emotion = 6
                
                # Copy image
                src = os.path.join(utkface_dir, filename)
                dst = os.path.join('data/images', filename)
                shutil.copy(src, dst)
                
                # Add label
                labels.append(f"{filename},{age},{emotion}\n")
                count += 1
                
                if count % 1000 == 0:
                    print(f"Processed {count} images...")
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    # Write labels file
    with open('data/labels.txt', 'w') as f:
        f.writelines(labels)
    
    print(f"\n✓ Converted {count} images")
    print(f"✓ Created labels.txt with {count} entries")
    print("\nDataset ready for training!")
    print("Run: python train.py")

if __name__ == "__main__":
    print("Face Dataset Preparation Tool")
    print("="*60)
    
    print("\nWhat would you like to do?")
    print("1. Create empty data structure")
    print("2. View dataset download instructions")
    print("3. Convert UTKFace dataset")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        create_data_structure()
    elif choice == '2':
        download_sample_dataset()
    elif choice == '3':
        utkface_path = input("Enter path to UTKFace dataset directory: ").strip()
        convert_utkface_dataset(utkface_path)
    else:
        print("Invalid choice!")
        print("\nRun this script again and choose 1, 2, or 3")
