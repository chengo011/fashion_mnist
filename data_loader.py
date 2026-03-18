import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from config import CLASS_NAMES

def load_data():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    return (X_train, y_train), (X_test, y_test)

def normalize(X_train, X_test):
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    return X_train, X_test

def flatten(X_train, X_test):
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    return X_train_flat, X_test_flat

def prepare_for_ml(X_train, X_test):
    X_train_norm, X_test_norm = normalize(X_train, X_test)
    X_train_flat, X_test_flat = flatten(X_train_norm, X_test_norm)
    return X_train_flat, X_test_flat

def prepare_for_cnn(X_train, X_test):
    X_train_norm, X_test_norm = normalize(X_train, X_test)

    X_train_cnn = X_train_norm[..., np.newaxis]
    X_test_cnn = X_test_norm[..., np.newaxis]

    return X_train_cnn, X_test_cnn

