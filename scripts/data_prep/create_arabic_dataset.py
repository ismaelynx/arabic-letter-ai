import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os
import random

arabic_letters = [
    'ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د',
    'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط',
    'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م',
    'ن', 'ه', 'و', 'ي'
]

def create_better_arabic_letter(letter, size=32):
    """Create properly sized and positioned Arabic letter"""
    
    # Try multiple font sizes to find the best one
    font_sizes = [16, 18, 20, 22, 24, 26, 28]
    best_img = None
    best_coverage = 0
    
    for font_size in font_sizes:
        # Create image with padding
        img = Image.new('L', (size, size), color=255)  # White background
        draw = ImageDraw.Draw(img)
        
        try:
            # Try to load a font
            font_paths = [
                '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
                '/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf',
            ]
            
            font = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        font = ImageFont.truetype(font_path, font_size)
                        break
                    except:
                        continue
            
            if font is None:
                font = ImageFont.load_default()
            
            # Get text dimensions
            bbox = draw.textbbox((0, 0), letter, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Skip if text is too big or too small
            if text_width <= 0 or text_height <= 0:
                continue
            if text_width > size * 0.9 or text_height > size * 0.9:
                continue
            if text_width < 4 or text_height < 4:
                continue
            
            # Center the text
            x = (size - text_width) // 2 - bbox[0]
            y = (size - text_height) // 2 - bbox[1]
            
            # Draw the letter
            draw.text((x, y), letter, fill=0, font=font)  # Black text
            
            # Check coverage (how much of image is used)
            img_array = np.array(img)
            black_pixels = np.sum(img_array < 200)  # Count non-white pixels
            total_pixels = size * size
            coverage = black_pixels / total_pixels
            
            # We want good coverage but not too much
            if 0.05 < coverage < 0.4 and coverage > best_coverage:
                best_coverage = coverage
                best_img = img.copy()
                
        except Exception as e:
            continue
    
    # If no good image found, create a simple geometric pattern
    if best_img is None:
        best_img = create_fallback_pattern(letter, size)
    
    return best_img

def create_fallback_pattern(letter, size):
    """Create simple geometric patterns if font fails"""
    img = Image.new('L', (size, size), color=255)
    draw = ImageDraw.Draw(img)
    
    # Get letter index
    try:
        letter_idx = arabic_letters.index(letter)
    except:
        letter_idx = 0
    
    # Create unique patterns based on letter index
    patterns = {
        0: lambda: draw.line([(size//2, 4), (size//2, size-4)], fill=0, width=2),  # ا - vertical line
        1: lambda: [draw.line([(4, size//2), (size-4, size//2)], fill=0, width=2),  # ب - horizontal + dot
                   draw.ellipse([(size//2-2, size//2+4), (size//2+2, size//2+8)], fill=0)],
        2: lambda: [draw.line([(4, size//2), (size-4, size//2)], fill=0, width=2),  # ت - horizontal + 2 dots
                   draw.ellipse([(size//2-4, size//2-8), (size//2-1, size//2-5)], fill=0),
                   draw.ellipse([(size//2+1, size//2-8), (size//2+4, size//2-5)], fill=0)],
    }
    
    # Use pattern if available, otherwise create based on index
    if letter_idx in patterns:
        result = patterns[letter_idx]()
    else:
        # Create unique pattern based on index
        if letter_idx % 4 == 0:
            draw.line([(4, 4), (size-4, size-4)], fill=0, width=2)
        elif letter_idx % 4 == 1:
            draw.ellipse([(4, 4), (size-4, size-4)], outline=0, width=2)
        elif letter_idx % 4 == 2:
            draw.rectangle([(4, 4), (size-4, size-4)], outline=0, width=2)
        else:
            draw.arc([(4, 4), (size-4, size-4)], 0, 180, fill=0, width=2)
    
    return img

def test_all_letters():
    """Test rendering of all Arabic letters"""
    print("Testing all Arabic letters...")
    
    fig, axes = plt.subplots(4, 7, figsize=(14, 8))
    fig.suptitle('Arabic Letters Test - Check if readable')
    
    results = []
    
    for i, letter in enumerate(arabic_letters):
        row = i // 7
        col = i % 7
        
        img = create_better_arabic_letter(letter)
        img_array = np.array(img)
        
        # Analyze the image
        black_pixels = np.sum(img_array < 200)
        total_pixels = 32 * 32
        coverage = black_pixels / total_pixels
        
        # Classify result
        if coverage < 0.01:
            status = "EMPTY"
        elif coverage > 0.8:
            status = "BLACK"
        elif coverage < 0.05:
            status = "TINY"
        else:
            status = "GOOD"
        
        results.append((letter, status, coverage))
        
        axes[row, col].imshow(img_array, cmap='gray')
        axes[row, col].set_title(f'{letter}\n{status}', fontsize=8)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('letter_test_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\nResults summary:")
    good_count = 0
    for letter, status, coverage in results:
        print(f"{letter}: {status} (coverage: {coverage:.3f})")
        if status == "GOOD":
            good_count += 1
    
    print(f"\n{good_count}/28 letters rendered properly")
    
    if good_count >= 20:
        print("✅ Most letters look good! Creating dataset...")
        return True
    else:
        print("❌ Too many letters have problems. Using fallback patterns...")
        return False

def create_dataset_with_working_letters():
    """Create dataset using the working approach"""
    
    # Test first
    font_working = test_all_letters()
    
    print(f"\nCreating dataset (font_working: {font_working})...")
    
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    
    samples_per_class = 100
    
    for class_idx, letter in enumerate(arabic_letters):
        print(f"Creating {letter} (class {class_idx})... ", end="")
        
        for i in range(samples_per_class):
            # Create letter
            img = create_better_arabic_letter(letter)
            
            # Add slight variations
            if random.random() > 0.5:
                # Rotate slightly
                angle = random.uniform(-10, 10)
                img = img.rotate(angle, fillcolor=255)
            
            # Add noise
            img_array = np.array(img).astype(float)
            noise = np.random.normal(0, 5, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            
            # Split train/test
            if i < 80:
                train_images.append(img_array)
                train_labels.append(class_idx)
            else:
                test_images.append(img_array)
                test_labels.append(class_idx)
        
        print("✓")
    
    # Save dataset
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    
    os.makedirs('./AHCD', exist_ok=True)
    np.save('../../datasets/original/train_images.npy', train_images)
    np.save('../../datasets/original/train_labels.npy', train_labels)
    np.save('../../datasets/original/test_images.npy', test_images)
    np.save('../../datasets/original/test_labels.npy', test_labels)
    
    print(f"\n✅ Dataset created!")
    print(f"Training samples: {len(train_images)}")
    print(f"Test samples: {len(test_images)}")
    
    # Show final samples
    fig, axes = plt.subplots(4, 7, figsize=(14, 8))
    fig.suptitle('Final Dataset - Ready for Training')
    
    for i in range(28):
        row = i // 7
        col = i % 7
        
        class_samples = train_images[train_labels == i]
        if len(class_samples) > 0:
            axes[row, col].imshow(class_samples[0], cmap='gray')
            axes[row, col].set_title(f'{arabic_letters[i]}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('final_dataset.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    create_dataset_with_working_letters()
    print("\n" + "="*50)
    print("Now run: python ahcd_letter_guess.py")
    print("="*50)
