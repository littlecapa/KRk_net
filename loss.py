import torch
import torch.nn as nn
import logging

def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Variable) dimension batch_size x 32 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5, ..., 31]

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    num_examples = outputs.size()[0]
    return -torch.sum(outputs[range(num_examples), labels])/num_examples

def delta_loss_fn(outputs, labels):
  batch_size = outputs.size()[0]
  # Find the index with the highest probability in predictions
  max_index = torch.argmax(outputs, dim=1)
  delta = torch.abs(max_index - labels).float()
  delta.requires_grad = outputs.requires_grad
  return torch.sum(delta)/batch_size

