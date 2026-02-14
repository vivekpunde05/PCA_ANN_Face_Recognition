import numpy as np

def calculate_accuracy(y_test, y_pred):
    correct = np.sum(y_test == y_pred)
    accuracy = (correct / len(y_test)) * 100
    return accuracy
