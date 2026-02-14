import numpy as np
from sklearn.model_selection import train_test_split

from src.data_loader import load_dataset
from src.feature_extraction import apply_pca, apply_lda
from src.neural_network import train_mlp, predict
from src.evaluation import calculate_accuracy
from src.visualization import plot_gallery
import matplotlib.pyplot as plt

# Configuration

DATASET_PATH = "dataset/faces/"
H = 300
W = 300
N_COMPONENTS = 150

# Load Dataset
X, y, class_names, n_samples = load_dataset(DATASET_PATH, H, W)

print("Number of samples:", n_samples)
print("Total dataset size:")
print("n_samples:", n_samples)
print("n_features:", X.shape[1])
print("n_classes:", len(class_names))

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# PCA
pca, eigenfaces, X_train_pca, X_test_pca = apply_pca(
    X_train, X_test, N_COMPONENTS, H, W
)

eigenfaces_titles = [f"eigenface {i}" for i in range(len(eigenfaces))]
plot_gallery(eigenfaces, eigenfaces_titles, H, W)
plt.show()


# LDA
lda, X_train_lda, X_test_lda = apply_lda(
    X_train_pca, X_test_pca, y_train
)


# Train Neural Network
clf = train_mlp(X_train_lda, y_train)


# Prediction
y_pred, y_prob = predict(clf, X_test_lda)


# Evaluation
accuracy = calculate_accuracy(y_test, y_pred)
print("Accuracy:", accuracy)

# Visualization of Results
prediction_titles = []
for i in range(len(y_pred)):
    true_name = class_names[y_test[i]]
    pred_name = class_names[y_pred[i]]

    result = f"pred: {pred_name}, pr: {str(y_prob[i])[:4]}\ntrue: {true_name}"
    prediction_titles.append(result)

plot_gallery(X_test, prediction_titles, H, W)
plt.show()
