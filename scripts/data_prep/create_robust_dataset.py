import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import os
import random
from scipy import ndimage

def load_perfect_dataset():
    """Load your perfect original dataset"""
    print("Loading perfect original dataset...")
    
    train_images = np.load('../../datasets/original/AHCD/train_images.npy')
    train_labels = np.load('../../datasets/original/AHCD/train_labels.npy')
    test_images = np.load('../../datasets/original/AHCD/test_images.npy')
    test_labels = np.load('../../datasets/original/AHCD/test_labels.npy')
    
    print(f"Original training samples: {len(train_images)}")
    print(f"Original test samples: {len(test_images)}")
    
    return train_images, train_labels, test_images, test_labels

def safe_augment_image(img_array, augmentation_type='light'):
    """Apply SAFE augmentations that preserve letter structure"""
    
    img = Image.fromarray(img_array.astype(np.uint8))
    
    if augmentation_type == 'rotation':
        # Small rotation
        angle = random.uniform(-8, 8)
        img = img.rotate(angle, fillcolor=255)
    
    elif augmentation_type == 'brightness':
        # Brightness change
        enhancer = ImageEnhance.Brightness(img)
        factor = random.uniform(0.8, 1.2)
        img = enhancer.enhance(factor)
    
    elif augmentation_type == 'contrast':
        # Contrast change
        enhancer = ImageEnhance.Contrast(img)
        factor = random.uniform(0.8, 1.2)
        img = enhancer.enhance(factor)
    
    elif augmentation_type == 'blur':
        # Very light blur
        if random.random() > 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    elif augmentation_type == 'noise':
        # Light noise
        img_array = np.array(img).astype(float)
        noise = np.random.normal(0, 5, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255)
        img = Image.fromarray(img_array.astype(np.uint8))
    
    elif augmentation_type == 'shift':
        # Small pixel shift
        img_array = np.array(img)
        shift_x = random.randint(-2, 2)
        shift_y = random.randint(-2, 2)
        
        if shift_x != 0 or shift_y != 0:
            img_array = ndimage.shift(img_array, (shift_y, shift_x), cval=255)
        
        img = Image.fromarray(img_array.astype(np.uint8))
    
    elif augmentation_type == 'thickness_light':
        # MUCH MORE CONSERVATIVE thickness change
        img_array = np.array(img)
        
        # Only apply to some pixels, not all
        if random.random() > 0.7:  # Only 30% chance
            binary = img_array < 128
            
            # Use smaller structuring element for more subtle changes
            structure = np.ones((2, 2))  # Smaller than default (3,3)
            
            if random.random() > 0.5:
                # Make very slightly thicker - only 1 iteration with small structure
                binary = ndimage.binary_dilation(binary, structure=structure, iterations=1)
            else:
                # Make very slightly thinner - be extra careful
                # Only erode if there are enough pixels to begin with
                if np.sum(binary) > 20:  # Ensure letter won't disappear
                    binary = ndimage.binary_erosion(binary, structure=structure, iterations=1)
            
            img_array = (~binary * 255).astype(np.uint8)
            
            # Quality check - if too much change, revert to original
            original_pixels = np.sum(img < 128)
            new_pixels = np.sum(img_array < 128)
            
            # If change is too dramatic, use original
            if abs(new_pixels - original_pixels) > original_pixels * 0.3:  # Max 30% change
                img_array = np.array(img)
        
        img = Image.fromarray(img_array.astype(np.uint8))
    
    return np.array(img)

def test_fixed_augmentations():
    """Test the FIXED augmentations"""
    print("Testing FIXED safe augmentations...")
    
    # Load one sample from original data
    train_images, train_labels, _, _ = load_perfect_dataset()
    
    # Pick a few samples including thin letters
    test_indices = [0, 100, 200, 300, 400, 500]
    augmentation_types = ['rotation', 'brightness', 'contrast', 'blur', 'noise', 'shift', 'thickness_light']
    
    fig, axes = plt.subplots(len(test_indices), len(augmentation_types) + 1, figsize=(16, 10))
    fig.suptitle('FIXED Augmentation Test - Thickness Should Be Subtle Now!')
    
    for i, idx in enumerate(test_indices):
        original_img = train_images[idx]
        original_label = train_labels[idx]
        
        # Show original
        axes[i, 0].imshow(original_img, cmap='gray')
        axes[i, 0].set_title(f'Original\n(class {original_label})')
        axes[i, 0].axis('off')
        
        # Show augmentations
        for j, aug_type in enumerate(augmentation_types):
            try:
                aug_img = safe_augment_image(original_img, aug_type)
                axes[i, j + 1].imshow(aug_img, cmap='gray')
                
                if aug_type == 'thickness_light':
                    axes[i, j + 1].set_title(f'{aug_type}\n(FIXED)', color='red')
                else:
                    axes[i, j + 1].set_title(aug_type)
                axes[i, j + 1].axis('off')
            except Exception as e:
                axes[i, j + 1].text(0.5, 0.5, f'Error:\n{aug_type}', 
                                   ha='center', va='center', transform=axes[i, j + 1].transAxes)
                axes[i, j + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('fixed_augmentation_test_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    response = input("Do ALL augmentations look good now? Especially thickness? (y/n): ")
    return response.lower() == 'y'

def create_hybrid_robust_dataset_fixed():
    """Create hybrid dataset with FIXED thickness augmentation"""
    
    # Test augmentations first
    if not test_fixed_augmentations():
        print("❌ Augmentations still don't look good.")
        return
    
    print("✅ Creating hybrid robust dataset with FIXED augmentations...")
    
    # Load perfect original data
    orig_train_images, orig_train_labels, orig_test_images, orig_test_labels = load_perfect_dataset()
    
    # Start with original data
    train_images = list(orig_train_images)
    train_labels = list(orig_train_labels)
    test_images = list(orig_test_images)
    test_labels = list(orig_test_labels)
    
    print("Adding augmented versions...")
    
    # Use FIXED thickness augmentation
    augmentation_types = ['rotation', 'brightness', 'contrast', 'blur', 'noise', 'shift', 'thickness_light']
    
    # Add 2 augmented versions of each training sample
    successful_augmentations = 0
    failed_augmentations = 0
    
    for i, (img, label) in enumerate(zip(orig_train_images, orig_train_labels)):
        if i % 500 == 0:
            print(f"Processed {i}/{len(orig_train_images)} samples...")
        
        # Create 2 different augmentations
        for _ in range(2):
            aug_type = random.choice(augmentation_types)
            try:
                aug_img = safe_augment_image(img, aug_type)
                
                # Quality check - make sure it's still reasonable
                black_pixels = np.sum(aug_img < 200)
                coverage = black_pixels / (32 * 32)
                
                if 0.02 < coverage < 0.6:  # Reasonable coverage
                    train_images.append(aug_img)
                    train_labels.append(label)
                    successful_augmentations += 1
                else:
                    failed_augmentations += 1
            except:
                failed_augmentations += 1
                continue
    
    # Add 1 augmented version of each test sample
    for i, (img, label) in enumerate(zip(orig_test_images, orig_test_labels)):
        aug_type = random.choice(augmentation_types)
        try:
            aug_img = safe_augment_image(img, aug_type)
            
            # Quality check
            black_pixels = np.sum(aug_img < 200)
            coverage = black_pixels / (32 * 32)
            
            if 0.02 < coverage < 0.6:
                test_images.append(aug_img)
                test_labels.append(label)
                successful_augmentations += 1
            else:
                failed_augmentations += 1
        except:
            failed_augmentations += 1
            continue
    
    print(f"Successful augmentations: {successful_augmentations}")
    print(f"Failed augmentations: {failed_augmentations}")
    
    # Convert to numpy
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    
    print(f"Final training samples: {len(train_images)}")
    print(f"Final test samples: {len(test_images)}")
    
    # Save hybrid dataset
    os.makedirs('./AHCD_HYBRID_FIXED', exist_ok=True)
    np.save('../../datasets/robust/AHCD_HYBRID_FIXED/train_images.npy', train_images)
    np.save('../../datasets/robust/AHCD_HYBRID_FIXED/train_labels.npy', train_labels)
    np.save('../../datasets/robust/AHCD_HYBRID_FIXED/test_images.npy', test_images)
    np.save('../../datasets/robust/AHCD_HYBRID_FIXED/test_labels.npy', test_labels)
    
    print("✅ FIXED HYBRID dataset created!")
    
    # Show samples
    show_hybrid_samples(train_images, train_labels)

def show_hybrid_samples(train_images, train_labels):
    """Show samples from hybrid dataset"""
    
    arabic_letters = [
        'ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د',
        'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط',
        'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م',
        'ن', 'ه', 'و', 'ي'
    ]
    
    fig, axes = plt.subplots(4, 7, figsize=(14, 8))
    fig.suptitle('FIXED Hybrid Dataset - All Augmentations Should Look Good!')
    
    for class_idx in range(28):
        row = class_idx // 7
        col = class_idx % 7
        
        # Get samples of this class
        class_samples = train_images[train_labels == class_idx]
        
        if len(class_samples) >= 4:
            # Show 4 different samples (mix of original and augmented)
            indices = np.random.choice(len(class_samples), 4, replace=False)
            
            # Create 2x2 grid for this class
            mini_grid = np.zeros((64, 64))
            mini_grid[0:32, 0:32] = class_samples[indices[0]]
            mini_grid[0:32, 32:64] = class_samples[indices[1]]
            mini_grid[32:64, 0:32] = class_samples[indices[2]]
            mini_grid[32:64, 32:64] = class_samples[indices[3]]
            
            axes[row, col].imshow(mini_grid, cmap='gray')
            axes[row, col].set_title(f'{arabic_letters[class_idx]}')
        
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('fixed_hybrid_dataset_samples_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    print("🔧 Creating FIXED HYBRID ROBUST Dataset")
    print("="*50)
    print("Fix: Much more conservative thickness changes")
    print("="*50)
    
    # Install scipy if needed
    try:
        from scipy import ndimage
    except ImportError:
        print("Installing scipy...")
        os.system("pip install scipy")
    
    create_hybrid_robust_dataset_fixed()
    
    print("\n" + "="*60)
    print("✅ FIXED HYBRID dataset ready!")
    print("Update your training script to use '../../datasets/robust/AHCD_HYBRID_FIXED/'")
    print("Thickness augmentation is now much more subtle!")
    print("="*60)
