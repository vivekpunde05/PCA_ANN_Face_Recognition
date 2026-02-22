# PCA ANN Face Recognition

A face recognition system using Principal Component Analysis (PCA), Linear Discriminant Analysis (LDA), and Artificial Neural Networks (ANN) for classification.

## Overview

This project implements a complete face recognition pipeline that:
- Loads and preprocesses face images from a dataset
- Applies PCA for dimensionality reduction and feature extraction
- Uses LDA for further feature optimization
- Trains a Multi-Layer Perceptron (MLP) neural network for classification
- Evaluates model performance and visualizes results

## Features

- **Dimensionality Reduction**: PCA reduces 90,000 features to 150 principal components
- **Feature Optimization**: LDA further refines features for better class separation
- **Neural Network Classification**: MLP with multiple hidden layers for accurate predictions
- **Visualization**: Display eigenfaces and prediction results with confidence scores
- **Multi-class Recognition**: Supports recognition of 9 different individuals

## Dataset

The dataset contains face images of 9 celebrities:
- Aamir Khan
- Ajay Devgn
- Akshay Kumar
- Alia Bhatt
- Amitabh Bachchan
- Deepika Padukone
- Disha Patani
- Farhan Akhtar
- Ileana D'Cruz

Total: 394 images (300x300 pixels)

## Requirements

```
numpy
opencv-python
matplotlib
scikit-learn
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/vivekpunde05/PCA_ANN_Face_Recognition.git
cd PCA_ANN_Face_Recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python main.py
```

The script will:
1. Load the face dataset
2. Split data into training (75%) and testing (25%) sets
3. Apply PCA to extract eigenfaces
4. Apply LDA for feature optimization
5. Train the MLP neural network
6. Display accuracy and visualizations

## Project Structure

```
PCA_ANN_Face_Recognition/
│
├── dataset/
│   ├── faces/          # Face images organized by person
│   └── Iris/           # Iris dataset (optional)
│
├── src/
│   ├── data_loader.py        # Dataset loading and preprocessing
│   ├── feature_extraction.py # PCA and LDA implementation
│   ├── neural_network.py     # MLP training and prediction
│   ├── evaluation.py         # Model evaluation metrics
│   └── visualization.py      # Result visualization
│
├── main.py             # Main execution script
├── requirements.txt    # Project dependencies
└── # Face Recognition using PCA, LDA, and Neural Networks

A machine learning project that implements face recognition using dimensionality reduction techniques (PCA and LDA) combined with a Multi-Layer Perceptron (MLP) neural network classifier.

## Overview

This project demonstrates a complete face recognition pipeline that:
- Loads and preprocesses face images from a dataset
- Applies Principal Component Analysis (PCA) for dimensionality reduction and eigenface extraction
- Uses Linear Discriminant Analysis (LDA) for further feature optimization
- Trains a Multi-Layer Perceptron neural network for classification
- Evaluates model performance and visualizes results

## Features

- **Data Loading**: Automatic loading of face images from organized directories
- **Preprocessing**: Image resizing and grayscale conversion
- **Eigenfaces**: Visualization of principal components as eigenfaces
- **Dimensionality Reduction**: PCA followed by LDA for optimal feature extraction
- **Neural Network Classification**: MLP classifier with configurable architecture
- **Performance Evaluation**: Accuracy calculation and prediction visualization
- **Results Visualization**: Gallery view of predictions with confidence scores

## Project Structure

```
.
├── main.py                      # Main execution script
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── dataset/
│   ├── faces/                   # Face images organized by person
│   │   ├── Aamir/
│   │   ├── Ajay/
│   │   ├── Akshay/
│   │   ├── Alia/
│   │   ├── Amitabh/
│   │   ├── Deepika/
│   │   ├── Disha/
│   │   ├── Farhan/
│   │   └── Ileana/
│   └── Iris/                    # Additional dataset
└── src/
    ├── data_loader.py           # Dataset loading utilities
    ├── feature_extraction.py    # PCA and LDA implementation
    ├── neural_network.py        # MLP training and prediction
    ├── evaluation.py            # Performance metrics
    └── visualization.py         # Plotting utilities
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <project-directory>
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies

- numpy: Numerical computing
- opencv-python: Image processing
- matplotlib: Visualization
- scikit-learn: Machine learning algorithms

## Usage

Run the main script to execute the complete face recognition pipeline:

```bash
python main.py
```

### Configuration

You can modify the following parameters in `main.py`:

- `DATASET_PATH`: Path to the face images directory (default: "dataset/faces/")
- `H`: Image height for resizing (default: 300)
- `W`: Image width for resizing (default: 300)
- `N_COMPONENTS`: Number of PCA components (default: 150)

### Dataset Structure

Organize your face images in the following structure:

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

Each subdirectory should contain images of a single person.

## How It Works

1. **Data Loading**: Images are loaded from the dataset directory, converted to grayscale, and resized to a uniform dimension.

2. **Train-Test Split**: The dataset is split into training (75%) and testing (25%) sets.

3. **PCA (Principal Component Analysis)**:
   - Reduces dimensionality from pixel space to 150 principal components
   - Extracts eigenfaces that capture the most variance in the data
   - Visualizes the top eigenfaces

4. **LDA (Linear Discriminant Analysis)**:
   - Further optimizes features for class separability
   - Transforms PCA features into a more discriminative space

5. **Neural Network Training**:
   - Multi-Layer Perceptron with two hidden layers (10, 10 neurons)
   - Trained on LDA-transformed features
   - Maximum 1000 iterations

6. **Prediction & Evaluation**:
   - Predicts identities for test images
   - Calculates prediction probabilities
   - Computes overall accuracy
   - Visualizes predictions with confidence scores

## Output

The program displays:
- Dataset statistics (samples, features, classes)
- Eigenfaces visualization
- Training progress
- Final accuracy score
- Gallery of test predictions with true labels and confidence scores

## Performance

The model's performance depends on:
- Dataset size and quality
- Number of PCA components
- Neural network architecture
- Training parameters

Typical accuracy ranges from 70-95% depending on dataset characteristics.

## Customization

### Modify Neural Network Architecture

Edit `src/neural_network.py`:
```python
clf = MLPClassifier(
    hidden_layer_sizes=(20, 20),  # Change layer sizes
    max_iter=2000,                # Increase iterations
    activation='relu',            # Change activation function
    solver='adam'                 # Change optimizer
)
```

### Adjust PCA Components

In `main.py`:
```python
N_COMPONENTS = 200  # Increase for more detail
```

### Change Image Dimensions

In `main.py`:
```python
H = 150  # Smaller for faster processing
W = 150
```

## Limitations

- Requires consistent lighting and face orientation in images
- Performance degrades with very small datasets
- Sensitive to image quality and preprocessing
- May struggle with significant pose variations

## Future Improvements

- Add data augmentation for better generalization
- Implement cross-validation for robust evaluation
- Support for real-time face recognition via webcam
- Deep learning alternatives (CNN-based approaches)
- Face detection preprocessing step
- Confusion matrix and detailed metrics

## License

[Specify your license here]

## Contributors

[Add contributor information]

## Acknowledgments

- Dataset: [Specify dataset source if applicable]
- Built with scikit-learn and OpenCV          # Project documentation
```

## Configuration

You can modify these parameters in `main.py`:

- `DATASET_PATH`: Path to face images directory
- `H`, `W`: Image height and width (default: 300x300)
- `N_COMPONENTS`: Number of PCA components (default: 150)

## Results

The model achieves approximately 69% accuracy on the test set with:
- 150 PCA components
- LDA transformation
- MLP with adaptive learning rate

## Visualization

The program displays two matplotlib windows:
1. **Eigenfaces**: Visual representation of principal components
2. **Predictions**: Test images with predicted vs. true labels and confidence scores

## License

This project is open source and available for educational purposes.

## Author

Vivek Punde

## Acknowledgments

- Dataset: Celebrity face images
- Libraries: scikit-learn, OpenCV, NumPy, Matplotlib
