import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from torch.utils.data import Dataset, DataLoader
import os
import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
FRAMES_PATH = r"D:\faceforensics_frames"  # Path to your frames
MODEL_PATH = "faceforensics_frame_model_final.pth"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("📊 DEEPFAKE DETECTION - PERFORMANCE METRICS")
print("=" * 70)
print(f"Device: {DEVICE}")
print(f"Model path: {MODEL_PATH}")
print(f"Frames path: {FRAMES_PATH}")
print("=" * 70)

# ============================================================================
# CNN MODEL DEFINITION (Must match your trained model)
# ============================================================================

class DeepfakeCNN(nn.Module):
    def __init__(self):
        super(DeepfakeCNN, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
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
# DATASET CLASS
# ============================================================================

class FaceFrameDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
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
# LOAD TEST DATA
# ============================================================================

def load_test_data():
    """Load test images from frames folder"""
    
    image_paths = []
    labels = []
    
    print("\n📂 Loading test data...")
    
    # Load REAL images (label = 0)
    real_folder = os.path.join(FRAMES_PATH, 'real')
    if os.path.exists(real_folder):
        real_images = glob.glob(os.path.join(real_folder, '*.jpg'))
        # Use 20% for testing (take last 20%)
        test_real = real_images[-int(len(real_images) * 0.2):]
        image_paths.extend(test_real)
        labels.extend([0] * len(test_real))
        print(f"✅ Loaded {len(test_real)} REAL test images")
    
    # Load FAKE images (label = 1)
    fake_folder = os.path.join(FRAMES_PATH, 'fake')
    if os.path.exists(fake_folder):
        fake_images = glob.glob(os.path.join(fake_folder, '*.jpg'))
        # Use 20% for testing
        test_fake = fake_images[-int(len(fake_images) * 0.2):]
        image_paths.extend(test_fake)
        labels.extend([1] * len(test_fake))
        print(f"✅ Loaded {len(test_fake)} FAKE test images")
    
    return image_paths, labels

# ============================================================================
# LOAD MODEL
# ============================================================================

def load_model():
    """Load trained model"""
    print("\n🔄 Loading trained model...")
    
    model = DeepfakeCNN()
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        model.to(DEVICE)
        model.eval()
        print(f"✅ Model loaded from {MODEL_PATH}")
        return model
    else:
        print(f"❌ Model not found at {MODEL_PATH}")
        return None

# ============================================================================
# PREDICT FUNCTION
# ============================================================================

def predict(model, dataloader):
    """Get predictions from model"""
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Predicting"):
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

# ============================================================================
# CALCULATE METRICS
# ============================================================================

def calculate_metrics(y_true, y_pred, y_probs):
    """Calculate all performance metrics"""
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred) * 100
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted') * 100
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted') * 100
    metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted') * 100
    
    # Per-class metrics
    metrics['precision_real'] = precision_score(y_true, y_pred, average=None)[0] * 100
    metrics['precision_fake'] = precision_score(y_true, y_pred, average=None)[1] * 100
    
    metrics['recall_real'] = recall_score(y_true, y_pred, average=None)[0] * 100
    metrics['recall_fake'] = recall_score(y_true, y_pred, average=None)[1] * 100
    
    metrics['f1_real'] = f1_score(y_true, y_pred, average=None)[0] * 100
    metrics['f1_fake'] = f1_score(y_true, y_pred, average=None)[1] * 100
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    # ROC and AUC
    if len(y_probs.shape) > 1:
        metrics['auc'] = auc(y_true, y_probs[:, 1]) * 100
    else:
        metrics['auc'] = auc(y_true, y_probs) * 100
    
    # Additional metrics
    tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
    metrics['specificity'] = tn / (tn + fp) * 100
    metrics['sensitivity'] = tp / (tp + fn) * 100
    metrics['false_positive_rate'] = fp / (fp + tn) * 100
    metrics['false_negative_rate'] = fn / (fn + tp) * 100
    
    return metrics

# ============================================================================
# PRINT METRICS TABLE
# ============================================================================

def print_metrics_table(metrics):
    """Print formatted metrics table"""
    
    print("\n" + "=" * 70)
    print("📊 MODEL PERFORMANCE METRICS")
    print("=" * 70)
    
    # Overall metrics
    print("\n📈 OVERALL METRICS:")
    print("-" * 50)
    print(f"  Accuracy:           {metrics['accuracy']:.2f}%")
    print(f"  Precision (weighted): {metrics['precision']:.2f}%")
    print(f"  Recall (weighted):    {metrics['recall']:.2f}%")
    print(f"  F1-Score (weighted):  {metrics['f1_score']:.2f}%")
    print(f"  AUC-ROC:            {metrics['auc']:.2f}%")
    
    # Per-class metrics
    print("\n🎯 PER-CLASS METRICS:")
    print("-" * 50)
    print(f"  {'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'REAL':<10} {metrics['precision_real']:.2f}%      {metrics['recall_real']:.2f}%      {metrics['f1_real']:.2f}%")
    print(f"  {'FAKE':<10} {metrics['precision_fake']:.2f}%      {metrics['recall_fake']:.2f}%      {metrics['f1_fake']:.2f}%")
    
    # Error rates
    print("\n⚠️ ERROR RATES:")
    print("-" * 50)
    print(f"  False Positive Rate: {metrics['false_positive_rate']:.2f}%")
    print(f"  False Negative Rate: {metrics['false_negative_rate']:.2f}%")
    print(f"  Specificity:         {metrics['specificity']:.2f}%")
    print(f"  Sensitivity:         {metrics['sensitivity']:.2f}%")

# ============================================================================
# PLOT CONFUSION MATRIX
# ============================================================================

def plot_confusion_matrix(cm, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['REAL', 'FAKE'], 
                yticklabels=['REAL', 'FAKE'])
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            plt.text(j+0.5, i+0.5, str(cm[i, j]), 
                    ha='center', va='center', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n✅ Confusion matrix saved to {save_path}")
    plt.show()

# ============================================================================
# PLOT ROC CURVE
# ============================================================================

def plot_roc_curve(y_true, y_probs, save_path='roc_curve.png'):
    """Plot ROC curve"""
    
    fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1] if len(y_probs.shape) > 1 else y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✅ ROC curve saved to {save_path}")
    plt.show()

# ============================================================================
# PLOT PRECISION-RECALL CURVE
# ============================================================================

def plot_pr_curve(y_true, y_probs, save_path='pr_curve.png'):
    """Plot Precision-Recall curve"""
    
    precision, recall, _ = precision_recall_curve(y_true, y_probs[:, 1] if len(y_probs.shape) > 1 else y_probs)
    ap = average_precision_score(y_true, y_probs[:, 1] if len(y_probs.shape) > 1 else y_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {ap:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✅ Precision-Recall curve saved to {save_path}")
    plt.show()

# ============================================================================
# PLOT METRICS BAR CHART
# ============================================================================

def plot_metrics_bar_chart(metrics, save_path='metrics_bar_chart.png'):
    """Plot bar chart of key metrics"""
    
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    values = [metrics['accuracy'], metrics['precision'], metrics['recall'], 
              metrics['f1_score'], metrics['auc']]
    colors = ['#2ecc71', '#3498db', '#3498db', '#3498db', '#9b59b6']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.ylim(0, 105)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.title('Model Performance Metrics', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✅ Metrics bar chart saved to {save_path}")
    plt.show()

# ============================================================================
# GENERATE REPORT
# ============================================================================

def generate_report(metrics, save_path='performance_report.txt'):
    """Generate text report"""
    
    with open(save_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("DEEPFAKE DETECTION - PERFORMANCE METRICS REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("MODEL INFORMATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Model Architecture: ResNet50 + Custom Classifier\n")
        f.write(f"Total Parameters: 24.55M\n")
        f.write(f"Input Size: 224x224x3\n")
        f.write(f"Output Classes: REAL (0), FAKE (1)\n\n")
        
        f.write("OVERALL METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy:                 {metrics['accuracy']:.2f}%\n")
        f.write(f"Precision (weighted):     {metrics['precision']:.2f}%\n")
        f.write(f"Recall (weighted):        {metrics['recall']:.2f}%\n")
        f.write(f"F1-Score (weighted):      {metrics['f1_score']:.2f}%\n")
        f.write(f"AUC-ROC:                  {metrics['auc']:.2f}%\n\n")
        
        f.write("PER-CLASS METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
        f.write(f"{'REAL':<10} {metrics['precision_real']:.2f}%       {metrics['recall_real']:.2f}%       {metrics['f1_real']:.2f}%\n")
        f.write(f"{'FAKE':<10} {metrics['precision_fake']:.2f}%       {metrics['recall_fake']:.2f}%       {metrics['f1_fake']:.2f}%\n\n")
        
        f.write("CONFUSION MATRIX\n")
        f.write("-" * 40 + "\n")
        cm = metrics['confusion_matrix']
        f.write(f"{'':<12} {'Predicted REAL':<15} {'Predicted FAKE':<15}\n")
        f.write(f"{'Actual REAL':<12} {cm[0,0]:<15} {cm[0,1]:<15}\n")
        f.write(f"{'Actual FAKE':<12} {cm[1,0]:<15} {cm[1,1]:<15}\n\n")
        
        f.write("ERROR RATES\n")
        f.write("-" * 40 + "\n")
        f.write(f"False Positive Rate: {metrics['false_positive_rate']:.2f}%\n")
        f.write(f"False Negative Rate: {metrics['false_negative_rate']:.2f}%\n")
        f.write(f"Specificity:         {metrics['specificity']:.2f}%\n")
        f.write(f"Sensitivity:         {metrics['sensitivity']:.2f}%\n")
    
    print(f"\n✅ Performance report saved to {save_path}")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main execution function"""
    
    # Load test data
    test_paths, test_labels = load_test_data()
    
    if len(test_paths) == 0:
        print("❌ No test data found!")
        return
    
    # Create dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = FaceFrameDataset(test_paths, test_labels, transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"\n📊 Test dataset size: {len(test_dataset)} images")
    
    # Load model
    model = load_model()
    
    if model is None:
        return
    
    # Get predictions
    y_pred, y_true, y_probs = predict(model, test_loader)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_probs)
    
    # Print metrics
    print_metrics_table(metrics)
    
    # Print confusion matrix
    print("\n📊 CONFUSION MATRIX:")
    print("-" * 40)
    cm = metrics['confusion_matrix']
    print(f"{'':<12} {'Predicted REAL':<15} {'Predicted FAKE':<15}")
    print(f"{'Actual REAL':<12} {cm[0,0]:<15} {cm[0,1]:<15}")
    print(f"{'Actual FAKE':<12} {cm[1,0]:<15} {cm[1,1]:<15}")
    
    # Print classification report
    print("\n📊 CLASSIFICATION REPORT:")
    print("-" * 40)
    print(classification_report(y_true, y_pred, target_names=['REAL', 'FAKE']))
    
    # Plot visualizations
    print("\n" + "=" * 70)
    print("📈 GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    plot_confusion_matrix(cm)
    plot_roc_curve(y_true, y_probs)
    plot_pr_curve(y_true, y_probs)
    plot_metrics_bar_chart(metrics)
    
    # Generate report
    generate_report(metrics)
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ PERFORMANCE METRICS COMPLETE!")
    print("=" * 70)
    print("\nFiles generated:")
    print("  1. confusion_matrix.png")
    print("  2. roc_curve.png")
    print("  3. pr_curve.png")
    print("  4. metrics_bar_chart.png")
    print("  5. performance_report.txt")
    print("=" * 70)

if __name__ == "__main__":
    main()
