import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2
import os
from PIL import Image
from models.multitask_model import MultiTaskFaceModel
from losses.multitask_loss import AdaptiveMultiTaskLoss, compute_multitask_loss
from losses.detection_loss import compute_detection_loss
from losses.age_loss import age_distribution_loss
from losses.emotion_loss import emotion_loss_fn

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Dataset class for face images with age and emotion labels
class FaceDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Dataset structure expected:
        data_dir/
            images/
                img1.jpg
                img2.jpg
            labels.txt (format: filename,age,emotion,x1,y1,x2,y2)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        # Load labels
        labels_file = os.path.join(data_dir, 'labels.txt')
        if os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        filename = parts[0]
                        age = int(parts[1])
                        emotion = int(parts[2])
                        
                        # Optional bounding box
                        bbox = None
                        if len(parts) >= 7:
                            bbox = [int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6])]
                        
                        self.samples.append({
                            'filename': filename,
                            'age': age,
                            'emotion': emotion,
                            'bbox': bbox
                        })
        
        print(f"Loaded {len(self.samples)} samples from {data_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        img_path = os.path.join(self.data_dir, 'images', sample['filename'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert age to bin (8 bins)
        age_bin = min(sample['age'] // 10, 7)  # 0-9, 10-19, ..., 70+
        
        # Apply transforms
        if self.transform:
            image = self.transform(Image.fromarray(image))
        
        return {
            'image': image,
            'age': age_bin,
            'emotion': sample['emotion'],
            'bbox': sample['bbox'] if sample['bbox'] else [0, 0, 224, 224]
        }

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Training function
def train_model(data_dir, num_epochs=50, batch_size=16, learning_rate=1e-4):
    """
    Train the multi-task face model
    
    Args:
        data_dir: Directory containing training data
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
    """
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found!")
        print("\nTo train the model, you need to:")
        print("1. Create a 'data' folder in your project directory")
        print("2. Inside 'data', create an 'images' folder with face images")
        print("3. Create a 'labels.txt' file with format:")
        print("   filename,age,emotion")
        print("   Example: face1.jpg,25,3")
        print("\nEmotion codes: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral")
        print("\nAlternatively, download a dataset like:")
        print("- UTKFace: https://susanqq.github.io/UTKFace/")
        print("- FER2013: https://www.kaggle.com/datasets/msambare/fer2013")
        print("- IMDB-WIKI: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/")
        return
    
    # Create dataset and dataloader
    try:
        dataset = FaceDataset(data_dir, transform=transform)
        
        if len(dataset) == 0:
            print("Error: No samples found in dataset!")
            return
        
        # Split into train and validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        print(f"Training samples: {train_size}, Validation samples: {val_size}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Initialize model, optimizer, and loss
    model = MultiTaskFaceModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = AdaptiveMultiTaskLoss()
    
    # Loss functions
    age_criterion = nn.CrossEntropyLoss()
    emotion_criterion = nn.CrossEntropyLoss()
    
    print("\nStarting training...")
    print("=" * 50)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_age_loss = 0.0
        train_emotion_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            age_labels = batch['age'].to(device)
            emotion_labels = batch['emotion'].to(device)
            
            # Create ROI proposals (using full image for now)
            batch_size = images.size(0)
            rois = torch.zeros((batch_size, 5), dtype=torch.float).to(device)
            for i in range(batch_size):
                rois[i] = torch.tensor([i, 0, 0, 224, 224])
            
            # Forward pass
            outputs = model(images, rois)
            
            # Compute losses using the improved loss function
            loss, age_loss, emotion_loss = compute_multitask_loss(
                outputs['age'], age_labels,
                outputs['emotion'], emotion_labels,
                lambda_age=1.0,
                lambda_emotion=1.0
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_age_loss += age_loss.item()
            train_emotion_loss += emotion_loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}, Age Loss: {age_loss.item():.4f}, "
                      f"Emotion Loss: {emotion_loss.item():.4f}")
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        avg_train_age_loss = train_age_loss / len(train_loader)
        avg_train_emotion_loss = train_emotion_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_age_loss = 0.0
        val_emotion_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                age_labels = batch['age'].to(device)
                emotion_labels = batch['emotion'].to(device)
                
                batch_size = images.size(0)
                rois = torch.zeros((batch_size, 5), dtype=torch.float).to(device)
                for i in range(batch_size):
                    rois[i] = torch.tensor([i, 0, 0, 224, 224])
                
                outputs = model(images, rois)
                
                # Compute losses using the improved loss function
                loss, age_loss, emotion_loss = compute_multitask_loss(
                    outputs['age'], age_labels,
                    outputs['emotion'], emotion_labels,
                    lambda_age=1.0,
                    lambda_emotion=1.0
                )
                
                val_loss += loss.item()
                val_age_loss += age_loss.item()
                val_emotion_loss += emotion_loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_age_loss = val_age_loss / len(val_loader)
        avg_val_emotion_loss = val_emotion_loss / len(val_loader)
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"Train Loss: {avg_train_loss:.4f} (Age: {avg_train_age_loss:.4f}, Emotion: {avg_train_emotion_loss:.4f})")
        print(f"Val Loss: {avg_val_loss:.4f} (Age: {avg_val_age_loss:.4f}, Emotion: {avg_val_emotion_loss:.4f})")
        print("=" * 50)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'model.pth')
            print(f"✓ Saved best model with validation loss: {best_val_loss:.4f}")
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("Model saved as 'model.pth'")

if __name__ == "__main__":
    print("Multi-Task Face Model Training Script")
    print("=" * 50)
    
    # Configuration
    DATA_DIR = 'data'  # Change this to your data directory
    NUM_EPOCHS = 50
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    
    # Start training
    train_model(DATA_DIR, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE)
