# Face Recognition using PCA, LDA, and Neural Networks

A machine learning project that implements face recognition using dimensionality reduction techniques (PCA and LDA) combined with a Multi-Layer Perceptron (MLP) neural network classifier.

## Overview

This project demonstrates a complete face recognition pipeline that:
- Loads and preprocesses face images from a dataset
- Applies Principal Component Analysis (PCA) for dimensionality reduction and eigenface extraction
- Uses Linear Discriminant Analysis (LDA) for further feature optimization
- Trains a Multi-Layer Perceptron neural network for classification
- Provides live webcam recognition and web interface
- Evaluates model performance and visualizes results

## Features

- **Data Loading**: Automatic loading of face images from organized directories
- **Preprocessing**: Image resizing and grayscale conversion
- **Eigenfaces**: Visualization of principal components as eigenfaces
- **Dimensionality Reduction**: PCA followed by LDA for optimal feature extraction
- **Neural Network Classification**: MLP classifier with configurable architecture
- **Performance Evaluation**: Accuracy calculation and prediction visualization
- **Results Visualization**: Gallery view of predictions with confidence scores
- **Live Webcam Recognition**: Real-time face detection and recognition ✨
- **Web Interface**: Interactive Streamlit web application for easy use ✨

## Project Structure

```
.
├── main.py                      # Main training script
├── live_recognition.py          # Real-time webcam face recognition ✨
├── web_app.py                   # Streamlit web application ✨
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── models/                      # Saved trained models (auto-created)
├── dataset/
│   └── faces/                   # Face images organized by person
│       ├── Aamir/
│       ├── Ajay/
│       ├── Akshay/
│       ├── Alia/
│       ├── Amitabh/
│       ├── Deepika/
│       ├── Disha/
│       ├── Farhan/
│       └── Ileana/
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
- opencv-python: Image processing and webcam access
- matplotlib: Visualization
- scikit-learn: Machine learning algorithms
- streamlit: Web application framework
- pillow: Image handling

## Usage

### 🚀 Quick Start - Web Application (Recommended)

Launch the interactive web interface:

```bash
streamlit run web_app.py
```

The web app will open in your browser at `http://localhost:8501`

**Features:**
- Upload images for recognition
- Train/retrain model from the interface
- View all recognized classes
- See detailed confidence scores
- User-friendly interface

### 📹 Live Webcam Recognition

For real-time face recognition using your webcam:

```bash
python live_recognition.py
```

**Features:**
- Automatically trains model if not found
- Real-time face detection and recognition
- Color-coded bounding boxes (green = recognized, orange = low confidence)
- Confidence scores displayed
- Press 'q' to quit

### 🎓 Train the Model (Command Line)

Run the main script to train the model and see visualizations:

```bash
python main.py
```

This will display eigenfaces and prediction results with matplotlib.

## Configuration

You can modify the following parameters in `main.py` or `live_recognition.py`:

- `DATASET_PATH`: Path to the face images directory (default: "dataset/faces/")
- `H`: Image height for resizing (default: 300)
- `W`: Image width for resizing (default: 300)
- `N_COMPONENTS`: Number of PCA components (default: 150)

## Dataset Structure

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

Each subdirectory should contain images of a single person. The folder name will be used as the person's label.

## How It Works

1. **Data Loading**: Images are loaded from the dataset directory, converted to grayscale, and resized to a uniform dimension (300x300).

2. **Train-Test Split**: The dataset is split into training (75%) and testing (25%) sets.

3. **PCA (Principal Component Analysis)**:
   - Reduces dimensionality from 90,000 features (300x300 pixels) to 150 principal components
   - Extracts eigenfaces that capture the most variance in the data
   - Visualizes the top eigenfaces

4. **LDA (Linear Discriminant Analysis)**:
   - Further optimizes features for class separability
   - Transforms PCA features into a more discriminative space

5. **Neural Network Training**:
   - Multi-Layer Perceptron with two hidden layers (10, 10 neurons)
   - Trained on LDA-transformed features
   - Maximum 1000 iterations with adaptive learning

6. **Face Detection (Live Mode)**:
   - Haar Cascade classifier detects faces in real-time
   - Each detected face is preprocessed and passed through the pipeline

7. **Prediction & Evaluation**:
   - Predicts identities for test images
   - Calculates prediction probabilities
   - Computes overall accuracy
   - Visualizes predictions with confidence scores

## Output

The program displays:
- Dataset statistics (samples, features, classes)
- Eigenfaces visualization (main.py)
- Training progress
- Final accuracy score
- Gallery of test predictions with true labels and confidence scores
- Real-time recognition results (live mode)

## Performance

The model's performance depends on:
- Dataset size and quality
- Number of PCA components
- Neural network architecture
- Training parameters
- Lighting conditions (for live recognition)

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

In `main.py` or `live_recognition.py`:
```python
N_COMPONENTS = 200  # Increase for more detail
```

### Change Image Dimensions

In `main.py` or `live_recognition.py`:
```python
H = 150  # Smaller for faster processing
W = 150
```

### Adjust Confidence Threshold

In `live_recognition.py`:
```python
recognizer.start_live_recognition(confidence_threshold=70)  # Higher = stricter
```

## Limitations

- Requires consistent lighting and face orientation in images
- Performance degrades with very small datasets
- Sensitive to image quality and preprocessing
- May struggle with significant pose variations
- Webcam recognition depends on camera quality and lighting

## Troubleshooting

### Webcam not working
- Ensure your webcam is connected and not being used by another application
- Try changing the camera index in `live_recognition.py`: `cv2.VideoCapture(1)` instead of `0`

### Low accuracy
- Increase the number of training images per person (recommended: 30+ images)
- Adjust PCA components
- Improve image quality and consistency
- Ensure proper lighting in training images

### Model not found
- Run `python main.py` first to train the model
- Or use the web app to train from the interface

## Future Improvements

- Add data augmentation for better generalization
- Implement cross-validation for robust evaluation
- ~~Support for real-time face recognition via webcam~~ ✅ Implemented
- Deep learning alternatives (CNN-based approaches)
- Face detection preprocessing step with MTCNN
- Confusion matrix and detailed metrics
- ~~Web interface for easy deployment~~ ✅ Implemented
- Mobile app version
- Multi-face tracking in video streams
- Face registration feature to add new people dynamically
- GPU acceleration for faster processing

## License

[Specify your license here]

## Contributors

[Add contributor information]

## Acknowledgments

- Dataset: Celebrity face images
- Built with scikit-learn, OpenCV, Streamlit, and NumPy
- Haar Cascade classifier from OpenCV for face detection
