# Face Recognition using PCA, LDA, and Neural Networks

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)

A machine learning project implementing face recognition using dimensionality reduction techniques (PCA and LDA) combined with a Multi-Layer Perceptron (MLP) neural network classifier.

## 📋 Table of Contents
- [Overview](#overview)
- [Demo & Results](#demo--results)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Performance Metrics](#performance-metrics)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🎯 Overview

This project demonstrates a complete face recognition pipeline that combines classical machine learning techniques with neural networks:

- **PCA (Principal Component Analysis)**: Reduces 90,000-dimensional image data to 150 principal components
- **LDA (Linear Discriminant Analysis)**: Optimizes features for class separability
- **MLP Neural Network**: Classifies faces with high accuracy
- **Eigenfaces Visualization**: Visual representation of principal components

## 🖼️ Demo & Results

### Sample Output

**Eigenfaces Visualization:**
The top principal components extracted from the face dataset, showing the most significant facial features.

**Prediction Results:**
- **Dataset**: 394 face images from 9 celebrities
- **Test Accuracy**: 67.68%
- **Features**: 90,000 → 150 (PCA) → 8 (LDA)
- **Training Time**: ~30 seconds on standard CPU

### Performance Metrics

| Metric | Value |
|--------|-------|
| Total Samples | 394 |
| Number of Classes | 9 |
| Train/Test Split | 75% / 25% |
| Original Features | 90,000 (300×300 pixels) |
| PCA Components | 150 |
| LDA Components | 8 |
| Test Accuracy | 67.68% |
| Neural Network Layers | [150, 10, 10, 9] |
| Training Iterations | 550 (early stopping) |

**Note**: Accuracy can be improved to 80-95% with:
- More training images per person (current: ~44 per class)
- Data augmentation
- Hyperparameter tuning
- Deeper neural network architecture

## ✨ Features

- ✅ **Automated Data Loading**: Loads face images from organized directories
- ✅ **Image Preprocessing**: Grayscale conversion and resizing
- ✅ **Eigenfaces Extraction**: Visualizes principal components
- ✅ **Dimensionality Reduction**: PCA + LDA pipeline
- ✅ **Neural Network Classification**: MLP with configurable architecture
- ✅ **Performance Evaluation**: Accuracy metrics and confidence scores
- ✅ **Results Visualization**: Gallery view with predictions
- ✅ **Modular Code Structure**: Clean, reusable components

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/PCA_ANN_Face_Recognition.git
cd PCA_ANN_Face_Recognition
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Dependencies
- `numpy` - Numerical computing
- `matplotlib` - Data visualization
- `scikit-learn` - Machine learning algorithms (PCA, LDA, MLP)

## 💻 Usage

### Train and Test the Model

Run the main script to train the model and visualize results:

```bash
python main.py
```

**Output:**
1. Dataset statistics
2. Eigenfaces visualization window
3. Training progress (loss per iteration)
4. Final accuracy score
5. Prediction gallery with confidence scores

### Dataset Structure

Organize your face images as follows:

```
dataset/faces/
├── Person1/
│   ├── face_1.jpg
│   ├── face_2.jpg
│   └── ...
├── Person2/
│   ├── face_1.jpg
│   └── ...
└── ...
```

Each subdirectory represents one person, and the folder name is used as the label.

## 🔬 How It Works

### Pipeline Architecture

```
Input Images (300×300) 
    ↓
Grayscale Conversion
    ↓
Flatten to Vector (90,000 features)
    ↓
PCA Reduction (150 components)
    ↓
LDA Optimization (8 components)
    ↓
MLP Neural Network (10, 10 hidden layers)
    ↓
Classification Output (9 classes)
```

### Detailed Steps

1. **Data Loading & Preprocessing**
   - Load images from dataset directory
   - Convert to grayscale
   - Resize to 300×300 pixels
   - Flatten to 90,000-dimensional vectors

2. **Train-Test Split**
   - 75% training data
   - 25% testing data
   - Stratified split to maintain class distribution

3. **PCA (Principal Component Analysis)**
   - Reduces dimensionality from 90,000 to 150
   - Captures 95%+ of variance
   - Generates eigenfaces for visualization
   - Speeds up training and reduces overfitting

4. **LDA (Linear Discriminant Analysis)**
   - Further reduces to 8 components (n_classes - 1)
   - Maximizes class separability
   - Optimizes features for classification

5. **Neural Network Training**
   - Architecture: [150, 10, 10, 9]
   - Activation: ReLU
   - Solver: Adam optimizer
   - Max iterations: 1000 (with early stopping)
   - Loss function: Cross-entropy

6. **Evaluation & Visualization**
   - Predict on test set
   - Calculate accuracy
   - Display predictions with confidence scores

## 📊 Performance Metrics

### Current Results

- **Accuracy**: 67.68% on test set
- **Training Loss**: Converged to 0.018 after 550 iterations
- **Inference Time**: <10ms per image

### Confusion Matrix Insights

The model performs well on:
- Faces with distinct features
- Well-lit, frontal images
- Consistent image quality

Challenges:
- Similar facial features between classes
- Varying lighting conditions
- Limited training samples per person

## 📁 Project Structure

```
PCA_ANN_Face_Recognition/
├── main.py                      # Main training and evaluation script
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
├── README.md                    # Project documentation
├── dataset/
│   ├── faces/                   # Face images organized by person
│   │   ├── Aamir/              # 50 images
│   │   ├── Ajay/               # 34 images
│   │   ├── Akshay/             # 50 images
│   │   ├── Alia/               # 50 images
│   │   ├── Amitabh/            # 25 images
│   │   ├── Deepika/            # 50 images
│   │   ├── Disha/              # 50 images
│   │   ├── Farhan/             # 35 images
│   │   └── Ileana/             # 50 images
│   └── Iris/                    # Additional dataset (not used)
└── src/
    ├── data_loader.py           # Dataset loading utilities
    ├── feature_extraction.py    # PCA and LDA implementation
    ├── neural_network.py        # MLP training and prediction
    ├── evaluation.py            # Performance metrics
    └── visualization.py         # Plotting utilities
```

## 🔧 Configuration

Modify parameters in `main.py`:

```python
# Dataset configuration
DATASET_PATH = "dataset/faces/"  # Path to face images
H = 300                          # Image height
W = 300                          # Image width

# PCA configuration
N_COMPONENTS = 150               # Number of principal components

# Neural Network (in src/neural_network.py)
hidden_layer_sizes = (10, 10)    # Hidden layer neurons
max_iter = 1000                  # Maximum training iterations
```

## 🚀 Future Improvements

### Planned Enhancements

- [ ] **Data Augmentation**: Rotation, flipping, brightness adjustment
- [ ] **Cross-Validation**: K-fold validation for robust evaluation
- [ ] **Hyperparameter Tuning**: Grid search for optimal parameters
- [ ] **Deep Learning**: CNN-based approach with transfer learning
- [ ] **Advanced Face Detection**: MTCNN or Dlib integration
- [ ] **Confusion Matrix**: Detailed per-class performance
- [ ] **Model Persistence**: Save/load trained models
- [ ] **Web Interface**: Flask/Streamlit deployment
- [ ] **Real-time Recognition**: Webcam integration
- [ ] **GPU Acceleration**: CUDA support for faster training

### Potential Applications

- Access control systems
- Attendance tracking
- Photo organization
- Security surveillance
- Social media tagging

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**Vivek Punde**
- Email: vivekpunde6@gmail.com
- LinkedIn: [linkedin.com/in/vivekpunde](https://www.linkedin.com/in/vivekpunde)
- GitHub: [@vivekpunde05](https://github.com/vivekpunde05)
- Portfolio: [vivekpunde.vercel.app](https://vivekpunde.vercel.app)

## 🙏 Acknowledgments

- **Dataset**: Celebrity face images from public sources
- **Libraries**: scikit-learn, NumPy, Matplotlib
- **Inspiration**: Eigenfaces paper by Turk and Pentland (1991)
- **Community**: Stack Overflow and GitHub contributors

---

⭐ **If you found this project helpful, please consider giving it a star!** ⭐

**Keywords**: `machine-learning` `face-recognition` `pca` `lda` `neural-networks` `python` `computer-vision` `eigenfaces` `dimensionality-reduction` `mlp` `scikit-learn`
