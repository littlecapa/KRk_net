import torch
import logging

def final_loss_fn(outputs, labels):
    
    # Convert log-probabilities to probabilities using exponentiation
    probabilities = torch.exp(outputs)
    
    # Extract the probability of the correct label for each example
    correct_label_probabilities = probabilities[range(outputs.size(0)), labels]
    
    # Compute 1 - probability of the correct label
    loss = 1 - correct_label_probabilities
    
    # Find the index of the highest probability for each example
    max_prob_indices = torch.argmax(probabilities, dim=1)
    
    # Create a binary tensor where 0 indicates correct label has highest probability, and 1 otherwise
    loss = torch.where(max_prob_indices != labels, torch.zeros_like(labels, dtype=torch.int), torch.ones_like(labels, dtype=torch.int))
    
    num_examples = outputs.size()[0]
    loss = torch.sum(loss)/num_examples
    return loss