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

