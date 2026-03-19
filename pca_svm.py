import numpy as np
import time
from sklearn.model_selection import train_test_split

from config import RANDOM_STATE
from data_loader import load_data, prepare_for_ml

(X_train, y_train), (X_test, y_test) = load_data()
X_train_flat, X_test_flat = prepare_for_ml(X_train, X_test)

X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train_flat, y_train,
    test_size=0.1,
    random_state=RANDOM_STATE
)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from config import PLOTS_DIR

pca_full = PCA().fit(X_train_split)

cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
os.makedirs(PLOTS_DIR, exist_ok=True)
plt.figure(figsize=(10, 5))
plt.plot(cumulative_variance)
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Schwelle')
plt.xlabel('Anzahl Komponenten')
plt.ylabel('Kumulierte erklärte Varianz')
plt.title('PCA – Explained Variance Ratio')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(PLOTS_DIR, 'pca_explained_variance.png'), dpi=150)
plt.close()
print(f"Plot gespeichert: {PLOTS_DIR}/pca_explained_variance.png")

n_components = np.argmax(cumulative_variance >= 0.95) + 1
print(f"\nKomponenten für 95% Varianz: {n_components}")
print(f"Erklärte Varianz mit {n_components} Komponenten: {cumulative_variance[n_components - 1]:.4f}")
