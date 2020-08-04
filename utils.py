import numpy as np

def precision_n(y_true, y_pred, n):
    res_idx = np.argsort(-y_pred)[:n]
    return sum(y_true.values[res_idx]) / n

def recall_n(y_true, y_pred, n):
    res_idx = np.argsort(-y_pred)[:n]
    return sum(y_true.values[res_idx]) / sum(y_true.values)

def precision_50(y_true, y_pred):
    res_idx = np.argsort(-y_pred)[:50]
    return sum(y_true.values[res_idx]) / 50