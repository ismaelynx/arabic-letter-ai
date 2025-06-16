import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os
import random

# Your model architecture (same as training script)
class ArabicCNN(nn.Module):
    def __init__(self, num_classes=28):
        super(ArabicCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
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
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

arabic_letters = [
    'ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د',
    'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط',
    'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م',
    'ن', 'ه', 'و', 'ي'
]

def load_robust_model():
    """Load the robust model"""
    model = ArabicCNN(num_classes=28)
    model.load_state_dict(torch.load('../../models/best_arabic_model_robust.pth', map_location='cpu'))
    model.eval()
    print("✅ Robust model loaded!")
    return model

def test_original_vs_robust():
    """Compare original dataset performance"""
    print("🧪 TEST: Original Dataset vs Robust Dataset Performance")
    
    model = load_robust_model()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Test on original dataset
    print("Testing on ORIGINAL dataset...")
    orig_test_images = np.load('../../datasets/original/test_images.npy')
    orig_test_labels = np.load('../../datasets/original/test_labels.npy')
    
    correct_orig = 0
    for img, label in zip(orig_test_images, orig_test_labels):
        pil_img = Image.fromarray(img.astype(np.uint8), mode='L')
        tensor_img = transform(pil_img).unsqueeze(0)
        
        with torch.no_grad():
            output = model(tensor_img)
            predicted = output.argmax(1).item()
        
        if predicted == label:
            correct_orig += 1
    
    orig_accuracy = correct_orig / len(orig_test_images) * 100
    
    # Test on robust dataset
    print("Testing on ROBUST dataset...")
    robust_test_images = np.load('../../datasets/robust/test_images.npy')
    robust_test_labels = np.load('../../datasets/robust/test_labels.npy')
    
    correct_robust = 0
    for img, label in zip(robust_test_images, robust_test_labels):
        pil_img = Image.fromarray(img.astype(np.uint8), mode='L')
        tensor_img = transform(pil_img).unsqueeze(0)
        
        with torch.no_grad():
            output = model(tensor_img)
            predicted = output.argmax(1).item()
        
        if predicted == label:
            correct_robust += 1
    
    robust_accuracy = correct_robust / len(robust_test_images) * 100
    
    print("="*60)
    print("📊 PERFORMANCE COMPARISON")
    print("="*60)
    print(f"Original Dataset:  {orig_accuracy:.2f}% ({correct_orig}/{len(orig_test_images)})")
    print(f"Robust Dataset:    {robust_accuracy:.2f}% ({correct_robust}/{len(robust_test_images)})")
    print("="*60)
    
    return orig_accuracy, robust_accuracy

def create_extreme_challenges():
    """Create extremely challenging test cases"""
    print("Creating extreme challenge cases...")
    
    # Load some original samples
    orig_images = np.load('../../datasets/original/test_images.npy')
    orig_labels = np.load('../../datasets/original/test_labels.npy')
    
    challenges = []
    test_indices = [0, 50, 100, 150, 200, 250, 300, 350]  # Different letters
    
    for idx in test_indices:
        img = orig_images[idx]
        label = orig_labels[idx]
        pil_img = Image.fromarray(img.astype(np.uint8), mode='L')
        
        # 1. Heavy rotation
        rotated = pil_img.rotate(20, fillcolor=255)
        challenges.append(('Heavy Rotation', np.array(rotated), label))
        
        # 2. Heavy blur
        blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=1.5))
        challenges.append(('Heavy Blur', np.array(blurred), label))
        
        # 3. Heavy noise
        img_array = np.array(pil_img).astype(float)
        noise = np.random.normal(0, 20, img_array.shape)
        noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        challenges.append(('Heavy Noise', noisy_array, label))
        
        # 4. Low contrast
        enhancer = ImageEnhance.Contrast(pil_img)
        low_contrast = enhancer.enhance(0.3)
        challenges.append(('Low Contrast', np.array(low_contrast), label))
    
    return challenges

def test_extreme_robustness():
    """Test extreme robustness"""
    print("🔥 EXTREME ROBUSTNESS TEST")
    
    model = load_robust_model()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    challenges = create_extreme_challenges()
    
    # Test each challenge
    fig, axes = plt.subplots(8, 4, figsize=(16, 20))
    fig.suptitle('🔥 EXTREME ROBUSTNESS TEST 🔥\nCan the Robust Model Handle These?', fontsize=16)
    
    results = []
    
    for i, (challenge_name, img, expected_label) in enumerate(challenges):
        row = i // 4
        col = i % 4
        
        # Predict
        pil_img = Image.fromarray(img.astype(np.uint8), mode='L')
        tensor_img = transform(pil_img).unsqueeze(0)
        
        with torch.no_grad():
            output = model(tensor_img)
            predicted = output.argmax(1).item()
            confidence = torch.softmax(output, dim=1).max().item() * 100
        
        is_correct = predicted == expected_label
        results.append(is_correct)
        
        # Display
        axes[row, col].imshow(img, cmap='gray')
        
        pred_letter = arabic_letters[predicted]
        expected_letter = arabic_letters[expected_label]
        
        if is_correct:
            color = 'green'
            title = f'✅ {pred_letter}\n{challenge_name}\n{confidence:.1f}%'
        else:
            color = 'red'
            title = f'❌ {pred_letter}\n(should be {expected_letter})\n{challenge_name}'
        
        axes[row, col].set_title(title, color=color, fontsize=9)
        axes[row, col].axis('off')
    
    extreme_accuracy = sum(results) / len(results) * 100
    
    plt.figtext(0.5, 0.02, f'EXTREME CHALLENGE ACCURACY: {extreme_accuracy:.1f}% ({sum(results)}/{len(results)})', 
                ha='center', fontsize=16, weight='bold', 
                color='green' if extreme_accuracy > 70 else 'orange' if extreme_accuracy > 50 else 'red')
    
    plt.tight_layout()
    plt.savefig('extreme_robustness_test_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"🔥 EXTREME CHALLENGE RESULTS: {extreme_accuracy:.1f}% ({sum(results)}/{len(results)})")
    
    return extreme_accuracy

def run_complete_robustness_suite():
    """Run the complete robustness test suite"""
    print("🚀 COMPLETE ROBUSTNESS TEST SUITE")
    print("="*70)
    
    # Test 1: Original vs Robust dataset
    orig_acc, robust_acc = test_original_vs_robust()
    
    print("\n" + "="*70)
    
    # Test 2: Extreme challenges
    extreme_acc = test_extreme_robustness()
    
    # Final summary
    print("\n" + "="*70)
    print("🏆 FINAL ROBUSTNESS REPORT")
    print("="*70)
    print(f"📊 Original Dataset Performance:  {orig_acc:.1f}%")
    print(f"🔧 Robust Dataset Performance:    {robust_acc:.1f}%")
    print(f"🔥 Extreme Challenge Performance: {extreme_acc:.1f}%")
    print("="*70)
    
    # Overall assessment
    if orig_acc > 95 and extreme_acc > 70:
        print("🎉 EXCELLENT: Model is both accurate AND robust!")
    elif orig_acc > 95 and extreme_acc > 50:
        print("👍 GOOD: High accuracy with decent robustness")
    elif orig_acc > 90:
        print("⚠️  OKAY: Good accuracy but limited robustness")
    else:
        print("❌ NEEDS WORK: Lower accuracy, may need more training")
    
    print("="*70)

if __name__ == '__main__':
    print("🧪 ROBUSTNESS TESTING SUITE")
    print("Wait for training to complete, then run this!")
    
    if os.path.exists('../../models/best_arabic_model_robust.pth'):
        run_complete_robustness_suite()
    else:
        print("❌ No robust model found. Train first!")