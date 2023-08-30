import torch
import torch.nn as nn
import logging

def loss_fn(outputs, labels):
    batch_size = outputs.size()[0]
    return -torch.sum(outputs[range(batch_size), labels])/batch_size

def delta_loss_fn(outputs, labels):
  batch_size = outputs.size()[0]
  # Find the index with the highest probability in predictions
  max_index = torch.argmax(outputs, dim=1)
  delta = torch.abs(max_index - labels).float()
  delta.requires_grad = outputs.requires_grad
  return torch.sum(delta)/batch_size

def delta_loss_mean_fn(outputs, labels):
  batch_size = outputs.size()[0]
  # Find the index with the highest probability in predictions
  max_index = torch.argmax(outputs, dim=1)
  delta = torch.abs(max_index - labels).float()
  delta.requires_grad = outputs.requires_grad
  return torch.mean(delta)/batch_size