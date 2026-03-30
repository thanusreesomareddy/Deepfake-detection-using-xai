import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================
FRAMES_PATH = r"D:\faceforensics_frames"  # Path to extracted frames
BATCH_SIZE = 32
EPOCHS = 10  # Keep as 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
START_EPOCH = 8  # MANUALLY SET TO START FROM EPOCH 8 (0-indexed, so 8 = epoch 9)

print("=" * 60)
print("TRAINING ON FACEFORENSICS++ FRAMES")
print("=" * 60)
print(f"Frames path: {FRAMES_PATH}")
print(f"Device: {DEVICE}")
print(f"Starting from epoch: {START_EPOCH + 1}/10")
print("=" * 60)

# ============================================================================
# CUSTOM DATASET CLASS
# ============================================================================

class FaceFrameDataset(Dataset):
    """Dataset for face frames"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels  # 0 = REAL, 1 = FAKE
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ============================================================================
# LOAD FRAMES
# ============================================================================

def load_frames():
    """Load all extracted frames"""
    
    image_paths = []
    labels = []
    
    print("\n📂 Loading frames...")
    
    # Load REAL frames
    real_folder = os.path.join(FRAMES_PATH, 'real')
    if os.path.exists(real_folder):
        real_images = glob.glob(os.path.join(real_folder, '*.jpg'))
        image_paths.extend(real_images)
        labels.extend([0] * len(real_images))
        print(f"✅ Loaded {len(real_images)} REAL frames")
    
    # Load FAKE frames
    fake_folder = os.path.join(FRAMES_PATH, 'fake')
    if os.path.exists(fake_folder):
        fake_images = glob.glob(os.path.join(fake_folder, '*.jpg'))
        image_paths.extend(fake_images)
        labels.extend([1] * len(fake_images))
        print(f"✅ Loaded {len(fake_images)} FAKE frames")
    
    # Split data
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=0.3, random_state=42, stratify=labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    print(f"\n📊 Split:")
    print(f"   Train: {len(train_paths)} frames")
    print(f"   Validation: {len(val_paths)} frames")
    print(f"   Test: {len(test_paths)} frames")
    
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)

# ============================================================================
# CNN MODEL
# ============================================================================

class DeepfakeCNN(nn.Module):
    def __init__(self):
        super(DeepfakeCNN, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# ============================================================================
# TRAINING FUNCTION WITH MANUAL START
# ============================================================================

def train_model(model, train_loader, val_loader, start_epoch):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    # Add placeholder for previous epochs to maintain history
    for i in range(start_epoch):
        train_losses.append(0)
        val_losses.append(0)
        train_accs.append(83.37)  # Your epoch 8 accuracy
        val_accs.append(83.69)     # Your epoch 8 accuracy
    
    print(f"\n🏋️ RESUMING TRAINING FROM EPOCH {start_epoch + 1}/10...\n")
    
    for epoch in range(start_epoch, EPOCHS):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for images, labels in train_pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
            for images, labels in val_pbar:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"\n📊 Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        print("-" * 50)
        
        # Save checkpoint after each epoch (for future resumes)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_acc': train_acc,
            'val_acc': val_acc
        }
        torch.save(checkpoint, 'checkpoint.pth')
        print(f"💾 Checkpoint saved (epoch {epoch+1})")
        
        # Save best model
        if val_acc == max(val_accs):
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"⭐ New best model! Val Acc: {val_acc:.2f}%")
    
    return train_losses, val_losses, train_accs, val_accs

# ============================================================================
# MAIN
# ============================================================================

def main():
    # Load data
    train_data, val_data, test_data = load_frames()
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = FaceFrameDataset(*train_data, transform)
    val_dataset = FaceFrameDataset(*val_data, transform)
    test_dataset = FaceFrameDataset(*test_data, transform)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Create model
    model = DeepfakeCNN().to(DEVICE)
    
    # LOAD THE MODEL FROM EPOCH 8
    # Since we don't have a checkpoint file, we need to load the model that was training
    
    print(f"\n🔄 Manually starting from epoch {START_EPOCH + 1}/10")
    print("   (Using model as it was at epoch 8 - before interruption)")
    
    # If you have a saved model from epoch 8, load it here
    # If not, we'll use the current model (which might be from epoch 8 if you didn't restart)
    model_path = 'faceforensics_frame_model.pth'  # Check if this exists
    if os.path.exists(model_path):
        print(f"✅ Loading saved model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        print("⚠️ No saved model found. Using current model state.")
        print("   If this is wrong, the training will be incorrect!")
    
    # Train from epoch 8
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, START_EPOCH
    )
    
    # Save final model
    final_model_path = 'faceforensics_frame_model_final.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"\n✅ Final model saved as '{final_model_path}'")
    
    # Test
    print("\n🧪 Testing on test set...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_acc = 100. * correct / total
    print(f"\n🎯 Final Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()
