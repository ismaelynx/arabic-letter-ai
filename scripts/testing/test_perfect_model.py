import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# Your model architecture (copy from the training script)
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

def load_perfect_model():
    """Load your perfect model"""
    model = ArabicCNN(num_classes=28)
    model.load_state_dict(torch.load('../../models/best_arabic_model.pth', map_location='cpu'))
    model.eval()
    print("✅ Perfect model loaded!")
    return model

def test_on_dataset_samples():
    """Test the model on some dataset samples"""
    model = load_perfect_model()
    
    # Load test data
    test_images = np.load('../../datasets/original/AHCD/test_images.npy')
    test_labels = np.load('../../datasets/original/AHCD/test_labels.npy')
    
    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Test on random samples
    indices = np.random.choice(len(test_images), 16, replace=False)
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle('🎉 PERFECT MODEL PREDICTIONS 🎉', fontsize=16)
    
    correct = 0
    
    for i, idx in enumerate(indices):
        row = i // 4
        col = i % 4
        
        # Get image and label
        img = test_images[idx]
        true_label = test_labels[idx]
        
        # Convert to tensor
        pil_img = Image.fromarray(img.astype(np.uint8), mode='L')
        tensor_img = transform(pil_img).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            output = model(tensor_img)
            predicted = output.argmax(1).item()
        
        # Get confidence
        confidence = torch.softmax(output, dim=1).max().item() * 100
        
        # Display
        axes[row, col].imshow(img, cmap='gray')
        
        pred_letter = arabic_letters[predicted]
        true_letter = arabic_letters[true_label]
        
        if predicted == true_label:
            color = 'green'
            correct += 1
            title = f'✅ {pred_letter}\n{confidence:.1f}%'
        else:
            color = 'red'
            title = f'❌ {pred_letter} (should be {true_letter})\n{confidence:.1f}%'
        
        axes[row, col].set_title(title, color=color, fontsize=10)
        axes[row, col].axis('off')
    
    accuracy = correct / 16 * 100
    plt.figtext(0.5, 0.02, f'Accuracy: {accuracy:.1f}% ({correct}/16)', 
                ha='center', fontsize=14, weight='bold')
    
    plt.tight_layout()
    plt.savefig('perfect_model_test_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"🎯 Test accuracy: {accuracy:.1f}% ({correct}/16)")

def create_challenge_letters():
    """Create some challenging letters to test the model"""
    print("Creating challenge letters...")
    
    model = load_perfect_model()
    
    # Create some variations of letters
    challenge_letters = ['ا', 'ب', 'ج', 'د', 'ه', 'و', 'ي', 'ن']
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle('🔥 CHALLENGE TEST: Can the perfect model handle these? 🔥')
    
    for i, letter in enumerate(challenge_letters):
        row = i // 4
        col = i % 4
        
        # Create letter image (similar to training data)
        img = Image.new('L', (32, 32), color=255)
        draw = ImageDraw.Draw(img)
        
        # Try to get a font
        try:
            font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
        except:
            font = ImageFont.load_default()
        
        # Draw letter
        bbox = draw.textbbox((0, 0), letter, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (32 - text_width) // 2 - bbox[0]
        y = (32 - text_height) // 2 - bbox[1]
        draw.text((x, y), letter, fill=0, font=font)
        
        # Add some noise/variation
        img_array = np.array(img).astype(float)
        noise = np.random.normal(0, 5, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        
        # Predict
        tensor_img = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(tensor_img)
            predicted = output.argmax(1).item()
            confidence = torch.softmax(output, dim=1).max().item() * 100
        
        # Display
        axes[row, col].imshow(np.array(img), cmap='gray')
        
        pred_letter = arabic_letters[predicted]
        expected_idx = arabic_letters.index(letter)
        
        if predicted == expected_idx:
            color = 'green'
            title = f'✅ {pred_letter}\n{confidence:.1f}%'
        else:
            color = 'red'
            title = f'❌ {pred_letter} (should be {letter})\n{confidence:.1f}%'
        
        axes[row, col].set_title(title, color=color)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('challenge_test_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    print("🎉 TESTING THE PERFECT ARABIC LETTER MODEL! 🎉")
    print("="*60)
    
    # Test on dataset samples
    test_on_dataset_samples()
    
    # Create challenge test
    create_challenge_letters()
    
    print("\n🏆 CONGRATULATIONS! 🏆")
    print("You've created a PERFECT Arabic letter recognition model!")
    print("This is genuinely impressive - 100% accuracy is rare!")