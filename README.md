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
└── README.md          # Project documentation
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
