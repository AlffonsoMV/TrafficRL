"""
Deep Q-Network Models
====================
Neural network models for Deep Q-Learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger("TrafficRL.Models")

class DQN(nn.Module):
    """
    Deep Q-Network for traffic light control.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(DQN, self).__init__()
        
        # Input layer to first hidden layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Add batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        # Add dropout for regularization
        self.dropout1 = nn.Dropout(0.2)
        
        # First hidden layer to second hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Add batch normalization
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        # Add dropout
        self.dropout2 = nn.Dropout(0.2)
        
        # Second hidden layer to third hidden layer (smaller)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        # Add batch normalization
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        
        # Output layer
        self.fc4 = nn.Linear(hidden_dim // 2, output_dim)
        
        # Initialize weights using He initialization
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc4.weight)
        
    def forward(self, x):
        # Check if we need to handle batch size of 1
        if x.dim() == 1 or x.size(0) == 1:
            # Process through network without batch norm for single samples
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
        else:
            # Normal batch processing with batch normalization and dropout
            x = F.relu(self.bn1(self.fc1(x)))
            x = self.dropout1(x)
            x = F.relu(self.bn2(self.fc2(x)))
            x = self.dropout2(x)
            x = F.relu(self.bn3(self.fc3(x)))
            x = self.fc4(x)
            
        return x


class IntersectionDQN(nn.Module):
    """
    DQN model that processes each intersection individually and outputs per-intersection actions.
    This model applies the same network to each intersection's state independently.
    """
    def __init__(self, features_per_intersection, output_dim, hidden_dim=128):
        super(IntersectionDQN, self).__init__()
        
        self.features_per_intersection = features_per_intersection
        
        # Create a single DQN model to be applied to each intersection
        self.intersection_model = DQN(
            input_dim=features_per_intersection, 
            output_dim=output_dim,
            hidden_dim=hidden_dim
        )
        
    def forward(self, x):
        """
        Process state data for all intersections.
        
        Args:
            x: Batch of states, should be of shape [batch_size, num_intersections * features_per_intersection]
               or [num_intersections * features_per_intersection] for a single state
        
        Returns:
            Q-values for all intersections [batch_size, num_intersections, output_dim]
            or [num_intersections, output_dim] for a single state
        """
        # Handle different input shapes
        original_shape = x.shape
        is_single_state = (len(original_shape) == 1)
        
        if is_single_state:
            # Add batch dimension if single state
            x = x.unsqueeze(0)
            
        batch_size = x.shape[0]
        
        # Reshape to [batch_size * num_intersections, features_per_intersection]
        num_intersections = x.shape[1] // self.features_per_intersection
        x = x.view(batch_size, num_intersections, self.features_per_intersection)
        x = x.reshape(batch_size * num_intersections, self.features_per_intersection)
        
        # Pass through the model
        q_values = self.intersection_model(x)
        
        # Reshape back to [batch_size, num_intersections, output_dim]
        q_values = q_values.view(batch_size, num_intersections, -1)
        
        # Remove batch dimension if it was a single state
        if is_single_state:
            q_values = q_values.squeeze(0)
            
        return q_values