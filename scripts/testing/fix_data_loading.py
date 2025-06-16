import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class ArabicLetterDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get image as numpy array
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to PIL Image (uint8, grayscale)
        image = Image.fromarray(image.astype(np.uint8), mode='L')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Simple transforms that work
train_transform = transforms.Compose([
    transforms.ToTensor(),  # Converts PIL to tensor and scales to [0,1]
    transforms.Normalize(mean=[0.5], std=[0.5])  # Scale to [-1,1]
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=28):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8x8 -> 4x4
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

def test_data_loading():
    """Test the fixed data loading"""
    print("Testing fixed data loading...")
    
    # Load data
    train_images = np.load('../../datasets/original/train_images.npy')
    train_labels = np.load('../../datasets/original/train_labels.npy')
    
    print(f"Loaded {len(train_images)} training samples")
    
    # Create dataset
    train_dataset = ArabicLetterDataset(train_images, train_labels, train_transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # Test loading one batch
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"Batch shape: {data.shape}")
        print(f"Target shape: {target.shape}")
        print(f"Data range: {data.min():.3f} to {data.max():.3f}")
        print(f"Labels: {target}")
        
        # Show the batch
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        for i in range(4):
            # Denormalize for display
            img = data[i].squeeze()
            img = (img * 0.5) + 0.5  # Convert from [-1,1] to [0,1]
            
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'Label: {target[i].item()}')
            axes[i].axis('off')
        
        plt.suptitle('Fixed Data Loading Test')
        plt.tight_layout()
        plt.savefig('fixed_data_test.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Data loading works!")
        break
    
    return train_dataset

def quick_training_test():
    """Quick test to see if training works now"""
    print("\nRunning quick training test...")
    
    # Load data
    train_images = np.load('../../datasets/original/train_images.npy')
    train_labels = np.load('../../datasets/original/train_labels.npy')
    test_images = np.load('../../datasets/original/test_images.npy')
    test_labels = np.load('../../datasets/original/test_labels.npy')
    
    # Create datasets
    train_dataset = ArabicLetterDataset(train_images, train_labels, train_transform)
    test_dataset = ArabicLetterDataset(test_images, test_labels, test_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create model
    model = SimpleCNN(num_classes=28)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train for 2 epochs to test
    model.train()
    for epoch in range(2):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 20 == 0:
                acc = 100. * correct / total
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {acc:.2f}%')
        
        # Test accuracy
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                outputs = model(data)
                _, predicted = outputs.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()
        
        test_acc = 100. * test_correct / test_total
        train_acc = 100. * correct / total
        
        print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        model.train()
    
    if test_acc > 10:
        print("✅ Training is working! Accuracy is improving!")
        return True
    else:
        print("❌ Still having issues...")
        return False

if __name__ == '__main__':
    # Test data loading first
    test_data_loading()
    
    # Run quick training test
    success = quick_training_test()
    
    if success:
        print("\n" + "="*50)
        print("✅ FIXED! Now update your main training script")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("❌ Still need to debug further")
        print("="*50)