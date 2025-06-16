# 🔤 Arabic Letter Recognition with Deep Learning

A deep learning project that recognizes handwritten Arabic letters using Convolutional Neural Networks (CNN). The model can identify all 28 Arabic letters with high accuracy.

## 🌟 Features

- **28 Arabic Letters Recognition**: Supports all Arabic alphabet letters
- **High Accuracy**: Achieves 99%+ accuracy on test data
- **Real Handwriting Support**: Trained on real handwritten Arabic letters
- **Multiple Model Architectures**: From basic CNN to advanced improved models
- **Comprehensive Dataset Tools**: Scripts for data preparation and augmentation
- **Extensive Testing**: Robustness testing and evaluation tools

## 📊 Supported Arabic Letters

| Letter | Name | Letter | Name | Letter | Name | Letter | Name |
|--------|------|--------|------|--------|------|--------|------|
| ا | alef | ب | beh | ت | teh | ث | theh |
| ج | jeem | ح | hah | خ | khah | د | dal |
| ذ | thal | ر | reh | ز | zain | س | seen |
| ش | sheen | ص | sad | ض | dad | ط | tah |
| ظ | zah | ع | ain | غ | ghain | ف | feh |
| ق | qaf | ك | kaf | ل | lam | م | meem |
| ن | noon | ه | heh | و | waw | ي | yeh |

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch torchvision numpy matplotlib pillow scikit-learn opencv-python
```

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd arabic_letter_recog
```

### 2. Download Real Dataset

```bash
cd scripts/data_prep
python download_real_dataset.py
```

### 3. Train the Model

```bash
cd ../training
python train_improved_model.py
```

### 4. Test the Model

```bash
python test_model_accuracy.py
```

## 📁 Project Structure

```
arabic_letter_recog/
├── README.md
├── requirements.txt
├── models/                          # Trained model files
│   ├── best_improved_model.pth
│   ├── best_arabic_model_robust.pth
│   └── model_checkpoints/
├── data/                           # Dataset storage
│   ├── processed/                  # Processed datasets
│   ├── real_dataset/              # Real handwriting datasets
│   └── original/                  # Original/synthetic datasets
├── scripts/
│   ├── data_prep/                 # Data preparation scripts
│   │   ├── download_real_dataset.py
│   │   ├── create_arabic_dataset.py
│   │   ├── create_realistic_dataset.py
│   │   └── pathprep.py
│   ├── training/                  # Model training scripts
│   │   ├── train_improved_model.py
│   │   ├── ahcd_letter_guess.py
│   │   └── model_architectures.py
│   └── testing/                   # Testing and evaluation
│       ├── test_model_accuracy.py
│       ├── test_robustness_complete.py
│       ├── test_perfect_model.py
│       └── fix_data_loading.py
└── notebooks/                     # Jupyter notebooks (optional)
    ├── data_exploration.ipynb
    └── model_analysis.ipynb
```

## 🏗️ Model Architectures

### 1. Basic CNN (`ArabicLetterCNN`)
- Simple 4-layer CNN
- Good for initial testing
- ~85-90% accuracy

### 2. Improved CNN (`ImprovedArabicCNN`)
- Advanced architecture with batch normalization
- Dropout layers for regularization
- Adaptive pooling
- **99%+ accuracy**

### 3. Robust CNN (`ArabicCNN`)
- Optimized for real handwriting
- Multiple convolutional blocks
- Strong regularization

## 📈 Training Results

| Model | Dataset | Accuracy | Training Time |
|-------|---------|----------|---------------|
| Basic CNN | Synthetic | 85.2% | ~10 min |
| Improved CNN | Real (AHCD) | 99.3% | ~30 min |
| Robust CNN | Mixed | 97.8% | ~25 min |

## 🔧 Usage Examples

### Training a New Model

```python
from scripts.training.train_improved_model import train_model

# Train with default settings
model, accuracy = train_model()

# Train with custom parameters
model, accuracy = train_model(
    epochs=50,
    batch_size=64,
    learning_rate=0.001
)
```

### Loading and Using a Trained Model

```python
import torch
from scripts.training.model_architectures import ImprovedArabicCNN

# Load model
model = ImprovedArabicCNN(num_classes=28)
checkpoint = torch.load('models/best_improved_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict on new image
prediction = model(image_tensor)
```

### Data Preparation

```python
# Create synthetic dataset
from scripts.data_prep.create_arabic_dataset import create_dataset
train_images, train_labels, test_images, test_labels = create_dataset()

# Download real dataset
from scripts.data_prep.download_real_dataset import process_and_save_ahcd
success = process_and_save_ahcd()
```

## 📊 Dataset Information

### Real Dataset (AHCD)
- **Source**: Arabic Handwritten Characters Dataset
- **Samples**: 16,800 total (13,440 train + 3,360 test)
- **Writers**: Multiple real writers
- **Format**: 32x32 grayscale images
- **Quality**: Real handwritten letters

### Synthetic Dataset
- **Samples**: 56,000+ generated samples
- **Variations**: Rotation, scaling, noise, blur
- **Format**: 32x32 grayscale images
- **Quality**: Clean, consistent

## 🧪 Testing and Evaluation

### Model Accuracy Testing
```bash
cd scripts/testing
python test_model_accuracy.py
```

### Robustness Testing
```bash
python test_robustness_complete.py
```

### Performance Metrics
- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: Per-class performance
- **Confusion Matrix**: Detailed error analysis
- **F1-Score**: Balanced performance metric

## 🔍 Troubleshooting

### Common Issues

1. **Low Accuracy on Real Handwriting**
   - Solution: Use the real AHCD dataset for training
   - Ensure proper data preprocessing

2. **Model Not Loading**
   - Check file paths in scripts
   - Verify model architecture matches saved model

3. **CUDA Out of Memory**
   - Reduce batch size in training scripts
   - Use CPU training if necessary

4. **Dataset Download Issues**
   - Manual download from Kaggle
   - Check internet connection and API credentials

### Performance Tips

- **For Better Accuracy**: Train on real handwriting data
- **For Faster Training**: Use GPU acceleration
- **For Lower Memory**: Reduce batch size
- **For Robustness**: Use data augmentation

## 📚 References

- **AHCD Dataset**: [Arabic Handwritten Characters Dataset](https://www.kaggle.com/datasets/mloey1/ahcd1)
- **Research Paper**: "Arabic Handwritten Characters Recognition using Convolutional Neural Network"
- **PyTorch Documentation**: [PyTorch Official Docs](https://pytorch.org/docs/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- AHCD dataset creators for providing real handwriting data
- PyTorch team for the excellent deep learning framework
- Arabic language community for inspiration and support

## 📞 Contact

- **Author**: Ismaael mehdi
- **Email**: [ismaael.moo@gmail.com]
- **GitHub**: [Your GitHub Profile]

---

⭐ **Star this repository if you found it helpful!** ⭐

## 🔄 Recent Updates

- **v2.0**: Added support for real handwriting dataset (AHCD)
- **v1.5**: Improved model architecture with 99%+ accuracy
- **v1.0**: Initial release with basic CNN model

## 🎯 Future Plans

- [ ] Add support for Arabic words recognition
- [ ] Implement real-time drawing interface
- [ ] Add mobile app support
- [ ] Expand to other Arabic script languages
- [ ] Add data augmentation techniques
- [ ] Implement ensemble methods