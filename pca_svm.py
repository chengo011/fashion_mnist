import numpy as np
import time
from sklearn.model_selection import train_test_split

from config import RANDOM_STATE, CLASS_NAMES, PLOTS_DIR
from data_loader import load_data, prepare_for_ml

(X_train, y_train), (X_test, y_test) = load_data()
X_train_flat, X_test_flat = prepare_for_ml(X_train, X_test)

X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train_flat, y_train,
    test_size=0.1,
    random_state=RANDOM_STATE
)


#Test-Code to determine the number of PCA components needed to retain variance
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

pca_full = PCA().fit(X_train_split)

cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
os.makedirs(PLOTS_DIR, exist_ok=True)
plt.figure(figsize=(10, 5))
plt.plot(cumulative_variance)
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Boundary')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('PCA – Explained Variance Ratio')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(PLOTS_DIR, 'pca_explained_variance.png'), dpi=150)
plt.close()
print(f"Plot saved: {PLOTS_DIR}/pca_explained_variance.png")

n_components = np.argmax(cumulative_variance >= 0.95) + 1
print(f"\nComponents for 95% Varaince: {n_components}")
print(f"Explained Variance with {n_components} Components: {cumulative_variance[n_components - 1]:.4f}")

def pca_fit(X_train, n_components):
    mean = np.mean(X_train, axis=0)
    X_centered = X_train - mean
    n = X_centered.shape[0]
    C = (1/n-1) * X_centered.T @ X_centered
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    eig_sort = np.argsort(eigenvalues)[::-1]
    ew = eigenvalues[eig_sort]
    ev = eigenvectors[:,eig_sort]
    components = eigenvectors[:, :n_components]
    evr = eigenvalues / np.sum(eigenvalues)

    return mean, components, evr

def pca_transform(X, mean, components):
    X_centered = X - mean
    return X_centered @ components

mean, components, evr = pca_fit(X_train_split, n_components=187)

X_train_pca = pca_transform(X_train_split, mean, components)
X_val_pca = pca_transform(X_val, mean, components)
X_test_pca = pca_transform(X_test_flat, mean, components)

#SVM on PCA-transformed data
from sklearn.svm import SVC

svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=RANDOM_STATE)
svm.fit(X_train_pca, y_train_split)

#Accuracy on Test-Set
val_accuracy = svm.score(X_val_pca, y_val)
print(f"Validation-Accuracy: {val_accuracy:.4f}")

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Prediction on test-data
y_pred = svm.predict(X_test_pca)

# Accuracy
test_accuracy = svm.score(X_test_pca, y_test)
print(f"\nTest-Accuracy: {test_accuracy:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix – PCA + SVM (Accuracy: {test_accuracy:.2%})')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'pca_svm_confusion_matrix.png'), dpi=150)
plt.close()
print(f"Plot saved: {PLOTS_DIR}/pca_svm_confusion_matrix.png")