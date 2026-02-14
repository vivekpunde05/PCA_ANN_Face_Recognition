import numpy as np
from sklearn.neural_network import MLPClassifier

def train_mlp(X_train, y_train):
    clf = MLPClassifier(
        random_state=1,
        hidden_layer_sizes=(10, 10),
        max_iter=1000,
        verbose=True
    )
    clf.fit(X_train, y_train)
    return clf


def predict(clf, X_test):
    y_pred = []
    y_prob = []

    for test_face in X_test:
        prob = clf.predict_proba([test_face])[0]
        class_id = np.argmax(prob)

        y_pred.append(class_id)
        y_prob.append(np.max(prob))

    return np.array(y_pred), np.array(y_prob)
