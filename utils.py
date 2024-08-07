import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_accuracy(y_predicted, y_actual):
    return sum(y_predicted == y_actual) / len(y_actual)