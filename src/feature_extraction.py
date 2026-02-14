import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def apply_pca(X_train, X_test, n_components, h, w):
    pca = PCA(n_components=n_components,
              svd_solver='randomized',
              whiten=True).fit(X_train)

    eigenfaces = pca.components_.reshape((n_components, h, w))

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    return pca, eigenfaces, X_train_pca, X_test_pca


def apply_lda(X_train_pca, X_test_pca, y_train):
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_pca, y_train)

    X_train_lda = lda.transform(X_train_pca)
    X_test_lda = lda.transform(X_test_pca)

    return lda, X_train_lda, X_test_lda
