import os
import shutil
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


RAW_TRAIN_IMAGES = '../../datasets/robust/train_images.npy'
RAW_TRAIN_LABELS = '../../datasets/robust/train_labels.npy'
RAW_TEST_IMAGES = '../../datasets/robust/test_images.npy'
RAW_TEST_LABELS = '../../datasets/robust/test_labels.npy'

# Target folder to store images by class for ImageFolder loader
DATA_DIR = '../../datasets/processed/data_robust'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Arabic letters classes as folder names (28 classes)
classes = (
    'ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د',
    'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط',
    'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م',
    'ن', 'ه', 'و', 'ي'
)

class ArabicLetterDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert numpy array to PIL Image
        image = Image.fromarray(image.astype(np.uint8), mode='L')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class ArabicCNN(nn.Module):
    def __init__(self, num_classes=28):
        super(ArabicCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8x8 -> 4x4
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 4x4 -> 2x2
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

def prepare_folder_structure(base_dir):
    """Create folder structure for dataset"""
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    for c in classes:
        os.makedirs(os.path.join(base_dir, c), exist_ok=True)

def save_images(images, labels, target_dir):
    """Save images into class folders"""
    for idx, (img_arr, label) in enumerate(zip(images, labels)):
        class_name = classes[label]
        folder = os.path.join(target_dir, class_name)
        img = Image.fromarray(img_arr.astype(np.uint8))
        img_path = os.path.join(folder, f'{idx}_results.png')
        img.save(img_path)

def prepare_dataset():
    """Prepare dataset from numpy arrays"""
    print("Loading robust dataset...")
    
    # Load robust numpy data
    train_images = np.load(RAW_TRAIN_IMAGES)
    train_labels = np.load(RAW_TRAIN_LABELS)
    test_images = np.load(RAW_TEST_IMAGES)
    test_labels = np.load(RAW_TEST_LABELS)
    
    print(f"Training samples: {len(train_images)}")
    print(f"Test samples: {len(test_images)}")
    
    # Create folder structures
    print("Creating folder structure...")
    prepare_folder_structure(TRAIN_DIR)
    prepare_folder_structure(TEST_DIR)
    
    # Save images into class folders
    print("Saving training images...")
    save_images(train_images, train_labels, TRAIN_DIR)
    
    print("Saving testing images...")
    save_images(test_images, test_labels, TEST_DIR)
    
    print("✅ Robust dataset prepared!")

def show_sample_images():
    """Display sample images from each class"""
    print("Loading sample images...")
    
    train_images = np.load(RAW_TRAIN_IMAGES)
    train_labels = np.load(RAW_TRAIN_LABELS)
    
    # Create a figure to show samples
    fig, axes = plt.subplots(4, 7, figsize=(15, 8))
    fig.suptitle('Arabic Letters Dataset Samples (Robust Version)', fontsize=16)
    
    for i, class_name in enumerate(classes):
        row = i // 7
        col = i % 7
        
        # Find first sample of this class
        class_indices = np.where(train_labels == i)[0]
        if len(class_indices) > 0:
            sample_img = train_images[class_indices[0]]
            axes[row, col].imshow(sample_img, cmap='gray')
            axes[row, col].set_title(f'{class_name} (Class {i})', fontsize=10)
        
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('arabic_letters_samples_robust_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def train_model():
    """Train the Arabic letter recognition model"""
    print("🚀 Starting training with ROBUST dataset...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_images = np.load(RAW_TRAIN_IMAGES)
    train_labels = np.load(RAW_TRAIN_LABELS)
    test_images = np.load(RAW_TEST_IMAGES)
    test_labels = np.load(RAW_TEST_LABELS)
    
    # Data transforms with additional augmentation for training
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        # Add some extra augmentation during training
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create datasets
    train_dataset = ArabicLetterDataset(train_images, train_labels, transform=train_transform)
    test_dataset = ArabicLetterDataset(test_images, test_labels, transform=test_transform)
    
    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Initialize model
    model = ArabicCNN(num_classes=28).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    
    # Training loop
    num_epochs = 25
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    best_test_acc = 0.0
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total_train += target.size(0)
            correct_train += predicted.eq(target).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        # Calculate training accuracy
        train_acc = 100. * correct_train / total_train
        avg_loss = running_loss / len(train_loader)
        
        # Testing phase
        model.eval()
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                total_test += target.size(0)
                correct_test += predicted.eq(target).sum().item()
        
        test_acc = 100. * correct_test / total_test
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), '../../models/best_arabic_model_robust.pth')
            print(f'✅ New best model saved! Test accuracy: {test_acc:.2f}%')
        
        # Store metrics
        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Test Acc: {test_acc:.2f}%, Best Test Acc: {best_test_acc:.2f}%')
        print('-' * 60)
    
    print(f'🎉 Training completed! Best test accuracy: {best_test_acc:.2f}%')
    
    # Plot training curves
    plot_training_curves(train_losses, train_accuracies, test_accuracies)
    
    return model

def plot_training_curves(train_losses, train_accuracies, test_accuracies):
    """Plot training curves"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    ax2.plot(epochs, test_accuracies, 'r-', label='Test Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves_robust_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model():
    """Evaluate the trained model"""
    print("🔍 Evaluating robust model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load test data
    test_images = np.load(RAW_TEST_IMAGES)
    test_labels = np.load(RAW_TEST_LABELS)
    
    # Load model
    model = ArabicCNN(num_classes=28).to(device)
    model.load_state_dict(torch.load('../../models/best_arabic_model_robust.pth', map_location=device))
    model.eval()
    
    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Test dataset
    test_dataset = ArabicLetterDataset(test_images, test_labels, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Evaluate
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate accuracy
    accuracy = sum(p == t for p, t in zip(all_predictions, all_targets)) / len(all_targets) * 100
    print(f"✅ Overall Test Accuracy: {accuracy:.2f}%")
    
    # Classification report
    print("\n📊 Classification Report:")
    print(classification_report(all_targets, all_predictions, target_names=classes))
    
    # Confusion matrix
    plot_confusion_matrix(all_targets, all_predictions)
    
    return accuracy

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - Robust Arabic Letter Recognition')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_robust_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_individual_predictions():
    """Test individual predictions with visualization"""
    print("🔍 Testing individual predictions...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load test data
    test_images = np.load(RAW_TEST_IMAGES)
    test_labels = np.load(RAW_TEST_LABELS)
    
    # Load model
    model = ArabicCNN(num_classes=28).to(device)
    model.load_state_dict(torch.load('../../models/best_arabic_model_robust.pth', map_location=device))
    model.eval()
    
    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Select random samples
    num_samples = 20
    indices = np.random.choice(len(test_images), num_samples, replace=False)
    
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    fig.suptitle('Individual Predictions - Robust Model', fontsize=16)
    
    correct_predictions = 0
    
    for i, idx in enumerate(indices):
        row = i // 5
        col = i % 5
        
        # Get image and true label
        img = test_images[idx]
        true_label = test_labels[idx]
        true_class = classes[true_label]
        
        # Predict
        pil_img = Image.fromarray(img.astype(np.uint8), mode='L')
        tensor_img = transform(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(tensor_img)
            probabilities = torch.softmax(output, dim=1)
            predicted_label = output.argmax(1).item()
            confidence = probabilities.max().item() * 100
        
        predicted_class = classes[predicted_label]
        is_correct = predicted_label == true_label
        
        if is_correct:
            correct_predictions += 1
        
        # Display
        axes[row, col].imshow(img, cmap='gray')
        
        if is_correct:
            color = 'green'
            title = f'✅ {predicted_class}\n{confidence:.1f}%'
        else:
            color = 'red'
            title = f'❌ {predicted_class}\n(True: {true_class})\n{confidence:.1f}%'
        
        axes[row, col].set_title(title, color=color, fontsize=10)
        axes[row, col].axis('off')
    
    sample_accuracy = correct_predictions / num_samples * 100
    plt.figtext(0.5, 0.02, f'Sample Accuracy: {sample_accuracy:.1f}% ({correct_predictions}/{num_samples})', 
                ha='center', fontsize=14, weight='bold')
    
    plt.tight_layout()
    plt.savefig('individual_predictions_robust_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def predict_single_image(image_path):
    """Predict a single image"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = ArabicCNN(num_classes=28).to(device)
    model.load_state_dict(torch.load('../../models/best_arabic_model_robust.pth', map_location=device))
    model.eval()
    
    # Load and preprocess image
    img = Image.open(image_path).convert('L')
    img = img.resize((32, 32))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    tensor_img = transform(img).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(tensor_img)
        probabilities = torch.softmax(output, dim=1)
        predicted_label = output.argmax(1).item()
        confidence = probabilities.max().item() * 100
    
    predicted_class = classes[predicted_label]
    
    # Display result
    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(np.array(img), cmap='gray')
    plt.title('Input Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    top_5_probs, top_5_indices = torch.topk(probabilities[0], 5)
    top_5_classes = [classes[i] for i in top_5_indices]
    top_5_probs = top_5_probs.cpu().numpy() * 100
    
    plt.barh(range(5), top_5_probs)
    plt.yticks(range(5), top_5_classes)
    plt.xlabel('Confidence (%)')
    plt.title('Top 5 Predictions')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.show()
    
    print(f"Predicted: {predicted_class} (Confidence: {confidence:.2f}%)")
    return predicted_class, confidence

def main():
    """Main function"""
    print("🚀 Arabic Letter Recognition - ROBUST VERSION")
    print("="*60)
    
    # Check if robust dataset exists
    if not os.path.exists(RAW_TRAIN_IMAGES):
        print("❌ Robust dataset not found!")
        print("Please run 'create_hybrid_robust_dataset_fixed.py' first")
        return
    
    while True:
        print("\nChoose an option:")
        print("1. Prepare dataset (convert numpy to folders)")
        print("2. Show sample images")
        print("3. Train model")
        print("4. Evaluate model")
        print("5. Test individual predictions")
        print("6. Predict single image")
        print("7. Exit")
        
        choice = input("Enter your choice (1-7): ").strip()
        
        if choice == '1':
            prepare_dataset()
        
        elif choice == '2':
            show_sample_images()
        
        elif choice == '3':
            model = train_model()
            print("✅ Training completed!")
        
        elif choice == '4':
            if os.path.exists('../../models/best_arabic_model_robust.pth'):
                evaluate_model()
            else:
                print("❌ No trained model found. Please train first (option 3)")
        
        elif choice == '5':
            if os.path.exists('../../models/best_arabic_model_robust.pth'):
                test_individual_predictions()
            else:
                print("❌ No trained model found. Please train first (option 3)")
        
        elif choice == '6':
            if os.path.exists('../../models/best_arabic_model_robust.pth'):
                image_path = input("Enter image path: ").strip()
                if os.path.exists(image_path):
                    predict_single_image(image_path)
                else:
                    print("❌ Image file not found")
            else:
                print("❌ No trained model found. Please train first (option 3)")
        
        elif choice == '7':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == '__main__':
    main()
   
