import os
import shutil
import numpy as np
from PIL import Image

# Paths - adjust these to your AHCD extracted files
RAW_TRAIN_IMAGES = '../../datasets/original/train_images.npy'
RAW_TRAIN_LABELS = '../../datasets/original/train_labels.npy'
RAW_TEST_IMAGES = '../../datasets/original/test_images.npy'
RAW_TEST_LABELS = '../../datasets/original/test_labels.npy'

# Target folder to store images by class for ImageFolder loader
DATA_DIR = '../../datasets/processed/data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Arabic letters classes as folder names (28 classes)
classes = [
    'alef', 'beh', 'teh', 'theh', 'jeem', 'hah', 'khah', 'dal',
    'thal', 'reh', 'zain', 'seen', 'sheen', 'sad', 'dad', 'tah',
    'zah', 'ain', 'ghain', 'feh', 'qaf', 'kaf', 'lam', 'meem',
    'noon', 'heh', 'waw', 'yeh'
]

# Alternative: Use numbers if Arabic names cause issues
# classes = [f'class_{i:02d}' for i in range(28)]

def check_data_integrity():
    """Check if the loaded data makes sense"""
    if not all(os.path.exists(f) for f in [RAW_TRAIN_IMAGES, RAW_TRAIN_LABELS, RAW_TEST_IMAGES, RAW_TEST_LABELS]):
        print("Error: Some dataset files are missing!")
        return False
    
    # Load and check data
    train_images = np.load(RAW_TRAIN_IMAGES)
    train_labels = np.load(RAW_TRAIN_LABELS)
    test_images = np.load(RAW_TEST_IMAGES)
    test_labels = np.load(RAW_TEST_LABELS)
    
    print("Dataset Information:")
    print(f"  Train images shape: {train_images.shape}")
    print(f"  Train labels shape: {train_labels.shape}")
    print(f"  Test images shape: {test_images.shape}")
    print(f"  Test labels shape: {test_labels.shape}")
    
    # Check label distribution
    unique_train_labels = np.unique(train_labels)
    unique_test_labels = np.unique(test_labels)
    
    print(f"  Unique train labels: {len(unique_train_labels)} -> {unique_train_labels}")
    print(f"  Unique test labels: {len(unique_test_labels)} -> {unique_test_labels}")
    print(f"  Label range: {train_labels.min()} to {train_labels.max()}")
    
    # Check if we have the expected number of classes
    if len(unique_train_labels) < 28:
        print(f"WARNING: Expected 28 classes, but found only {len(unique_train_labels)}")
        print("This might be sample data or incorrectly prepared data.")
    
    # Show label distribution
    print("\nLabel distribution in training set:")
    for label in unique_train_labels:
        count = np.sum(train_labels == label)
        class_name = classes[label] if label < len(classes) else f"class_{label}"
        print(f"  Label {label} ({class_name}): {count} samples")
    
    return True

def prepare_folder_structure(base_dir):
    """Create folder structure for each class"""
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    
    os.makedirs(base_dir, exist_ok=True)
    
    # Create folders for all possible classes
    for i, class_name in enumerate(classes):
        class_dir = os.path.join(base_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

def save_images(images, labels, target_dir):
    """Save numpy arrays as images in class folders"""
    print(f"Saving {len(images)} images to {target_dir}...")
    
    # Count images per class
    class_counts = {}
    
    for idx, (img_arr, label) in enumerate(zip(images, labels)):
        # Ensure label is within valid range
        if label >= len(classes):
            print(f"Warning: Label {label} is out of range (max: {len(classes)-1}). Skipping image {idx}")
            continue
        
        class_name = classes[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        folder = os.path.join(target_dir, class_name)
        
        # Handle different image shapes
        if len(img_arr.shape) == 3:  # RGB image
            if img_arr.shape[2] == 1:  # Single channel stored as 3D
                img_arr = img_arr.squeeze()
            img = Image.fromarray(img_arr.astype(np.uint8))
        else:  # Grayscale image
            img = Image.fromarray(img_arr.astype(np.uint8), mode='L')
        
        # Use class count for filename to avoid conflicts
        img_path = os.path.join(folder, f'{class_counts[class_name]:05d}.png')
        img.save(img_path)
        
        if idx % 1000 == 0:
            print(f"  Saved {idx} images...")
    
    # Print final class distribution
    print(f"\nClass distribution in {target_dir}:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count} images")

def create_balanced_sample_dataset():
    """Create a balanced sample dataset with all 28 classes"""
    print("Creating balanced sample dataset...")
    
    np.random.seed(42)
    samples_per_class = 100
    
    # Create sample data for all 28 classes
    all_images = []
    all_labels = []
    
    for class_id in range(28):
        # Generate random images for this class
        class_images = np.random.randint(0, 256, (samples_per_class, 32, 32), dtype=np.uint8)
        class_labels = np.full(samples_per_class, class_id, dtype=np.int64)
        
        all_images.append(class_images)
        all_labels.append(class_labels)
    
    # Combine all classes
    train_images = np.vstack(all_images)
    train_labels = np.hstack(all_labels)
    
    # Create test set (20% of training data)
    test_size = len(train_images) // 5
    indices = np.random.permutation(len(train_images))
    
    test_images = train_images[indices[:test_size]]
    test_labels = train_labels[indices[:test_size]]
    train_images = train_images[indices[test_size:]]
    train_labels = train_labels[indices[test_size:]]
    
    # Create AHCD directory and save
    os.makedirs('./AHCD', exist_ok=True)
    np.save('../../datasets/original/train_images.npy', train_images)
    np.save('../../datasets/original/train_labels.npy', train_labels)
    np.save('../../datasets/original/test_images.npy', test_images)
    np.save('../../datasets/original/test_labels.npy', test_labels)
    
    print(f"Created balanced sample dataset:")
    print(f"  Training: {len(train_images)} images, {len(np.unique(train_labels))} classes")
    print(f"  Testing: {len(test_images)} images, {len(np.unique(test_labels))} classes")

def prepare_dataset():
    """Main function to prepare the dataset"""
    print("Arabic Letter Dataset Preparation")
    print("=" * 50)
    
    # Check if we need to create sample data
    if not all(os.path.exists(f) for f in [RAW_TRAIN_IMAGES, RAW_TRAIN_LABELS, RAW_TEST_IMAGES, RAW_TEST_LABELS]):
        print("Dataset files not found. Creating balanced sample dataset...")
        create_balanced_sample_dataset()
    
    # Check data integrity
    if not check_data_integrity():
        return
    
    # Ask user if they want to continue with unbalanced data
    train_labels = np.load(RAW_TRAIN_LABELS)
    unique_labels = len(np.unique(train_labels))
    
    if unique_labels < 28:
        print(f"\nWARNING: Your dataset has only {unique_labels} classes instead of 28!")
        choice = input("Do you want to:\n1. Continue with current data\n2. Create balanced sample data\nChoice (1/2): ").strip()
        
        if choice == '2':
            create_balanced_sample_dataset()
    
    # Load the data
    train_images = np.load(RAW_TRAIN_IMAGES)
    train_labels = np.load(RAW_TRAIN_LABELS)
    test_images = np.load(RAW_TEST_IMAGES)
    test_labels = np.load(RAW_TEST_LABELS)
    
    # Create folder structures
    print("\nCreating folder structure...")
    prepare_folder_structure(TRAIN_DIR)
    prepare_folder_structure(TEST_DIR)
    
    # Save images into class folders
    print("\nSaving training images...")
    save_images(train_images, train_labels, TRAIN_DIR)
    
    print("\nSaving testing images...")
    save_images(test_images, test_labels, TEST_DIR)
    
    print("\n" + "="*50)
    print("Dataset prepared successfully!")
    print(f"Training data: {TRAIN_DIR}")
    print(f"Testing data: {TEST_DIR}")
    print("="*50)

if __name__ == '__main__':
    prepare_dataset()
