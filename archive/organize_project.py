import os
import shutil

def organize_files():
    """Organize all files into clean structure"""
    print("🗂️  Organizing 34+ files...")
    
    # Create organized structure
    folders = [
        'datasets',
        'datasets/original', 
        'datasets/robust',
        'datasets/processed',
        'models',
        'scripts',
        'scripts/data_prep',
        'scripts/training', 
        'scripts/testing',
        'results',
        'results/plots',
        'results/training_curves',
        'results/evaluations',
        'archive'
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    
    # File organization mapping
    moves = {
        # === DATASETS ===
        'AHCD/': 'datasets/original/',
        'AHCD_HYBRID_FIXED/': 'datasets/robust/', 
        'data/': 'datasets/processed/',
        
        # === MODELS ===
        'best_arabic_model.pth': 'models/',
        'best_arabic_model_robust.pth': 'models/',
        'best_model.pth': 'models/',
        'interrupted_model.pth': 'archive/',
        
        # === MAIN SCRIPTS ===
        'ahcd_letter_guess.py': 'scripts/training/',
        'create_robust_dataset.py': 'scripts/data_prep/',
        'create_arabic_dataset.py': 'scripts/data_prep/',
        'pathprep.py': 'scripts/data_prep/',
        
        # === TESTING SCRIPTS ===
        'save_robustness_complete.py': 'scripts/testing/',
        'test_perfect_model.py': 'scripts/testing/',
        'debug_training.py': 'scripts/testing/',
        'fix_data_loading.py': 'scripts/testing/',
        
        # === SAMPLE/VERIFICATION IMAGES ===
        'arabic_letters_samples_robust.png': 'results/plots/',
        'real_ahcd_samples.png': 'results/plots/',
        'real_ahcd_verification.png': 'results/plots/',
        'final_arabic_dataset.png': 'results/plots/',
        'final_dataset.png': 'results/plots/',
        'fixed_hybrid_dataset_samples.png': 'results/plots/',
        'debug_samples.png': 'results/plots/',
        'class_averages.png': 'results/plots/',
        
        # === TEST RESULTS ===
        'augmentation_test.png': 'results/evaluations/',
        'challenge_test.png': 'results/evaluations/',
        'controlled_variation_test.png': 'results/evaluations/',
        'extreme_robustness_test.png': 'results/evaluations/',
        'font_diversity_test.png': 'results/evaluations/',
        'font_test.png': 'results/evaluations/',
        'perfect_model_test.png': 'results/evaluations/',
        'fixed_augmentation_test.png': 'results/evaluations/',
        'fixed_data_test.png': 'results/evaluations/',
        
        # === TRAINING RESULTS ===
        'training_curves.png': 'results/training_curves/',
        'confusion_matrix_robust.png': 'results/training_curves/',
        'individual_predictions_robust.png': 'results/training_curves/',
        
        # === COMPARISON IMAGES ===
        'letter_comparison.png': 'results/evaluations/',
        'letter_test_results.png': 'results/evaluations/',
        
        # === UTILITY ===
        'organize_project.py': 'archive/',
    }
    
    # Move files
    moved_count = 0
    for source, destination in moves.items():
        if os.path.exists(source):
            try:
                # Handle directories
                if os.path.isdir(source):
                    dest_path = os.path.join(destination, os.path.basename(source))
                    if os.path.exists(dest_path):
                        shutil.rmtree(dest_path)
                    shutil.move(source, destination)
                    print(f"📁 {source} → {destination}")
                # Handle files  
                else:
                    shutil.move(source, destination)
                    print(f"📄 {source} → {destination}")
                moved_count += 1
            except Exception as e:
                print(f"⚠️  Error moving {source}: {e}")
    
    print(f"\n✅ Organized {moved_count} items!")
    
    # Show final structure
    print("\n📁 FINAL STRUCTURE:")
    print("="*40)
    for root, dirs, files in os.walk('.'):
        if root == '.':
            continue
        level = root.replace('./', '').count('/')
        indent = '  ' * level
        folder_name = os.path.basename(root)
        file_count = len(files)
        print(f"{indent}📁 {folder_name}/ ({file_count} files)")

if __name__ == '__main__':
    organize_files()
