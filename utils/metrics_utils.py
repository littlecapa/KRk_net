import logging

import numpy as np
import torch

def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)

def test_ok(outputs, labels):
    outputs = np.argmax(outputs, axis=1)
    abs_diff = np.abs(outputs - labels)
    test_values = np.zeros_like(outputs)
    # Define conditions directly as a list of booleans
    condition = [
        abs_diff <= 2,
        (abs_diff <= 4) & (labels > 10) & (labels < 20),
        (abs_diff <= 8) & (labels > 20)
    ]

    for i in range(len(condition)):
        test_values[condition[i]] = 1.0
    return np.sum(test_values==1.0)/float(labels.size)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    'ok': test_ok,
    # could add more metrics such as accuracy for each token type
}

def metrics_to_csv_string(metrics):
    csv_string = ""
    for key, value in metrics.items():
        csv_string += f"{value}; "
    return csv_string.rstrip("; ")  # Remove the trailing semicolon and space