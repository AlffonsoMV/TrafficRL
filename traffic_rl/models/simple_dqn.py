"""
Simple Deep Q-Network Model
=========================
A basic implementation of DQN without advanced features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger("TrafficRL.Models")

class SimpleDQN(nn.Module):
    """
    Simple Deep Q-Network for traffic light control.
    Basic architecture with just two hidden layers.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(SimpleDQN, self).__init__()
        
        # Simple architecture with two hidden layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights using Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        
    def forward(self, x):
        # Simple forward pass with ReLU activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 