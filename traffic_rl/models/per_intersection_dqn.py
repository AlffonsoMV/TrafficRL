"""
Per-Intersection Deep Q-Network Model
==================================
Neural network model for controlling multiple intersections independently.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np

logger = logging.getLogger("TrafficRL.Models")

class PerIntersectionDQN(nn.Module):
    """
    Per-Intersection Deep Q-Network for traffic light control.
    
    This model processes each intersection's state independently and outputs 
    separate action values for each intersection.
    """
    def __init__(self, state_size, action_size, num_intersections, hidden_dim=64):
        super(PerIntersectionDQN, self).__init__()
        
        # Store dimensions
        self.state_size = state_size
        self.action_size = action_size
        self.num_intersections = num_intersections
        
        # Calculate features per intersection
        self.features_per_intersection = 5  # Based on observation space
        
        # Shared feature extraction layers
        self.shared_fc1 = nn.Linear(self.features_per_intersection, hidden_dim)
        self.shared_bn1 = nn.BatchNorm1d(hidden_dim)
        
        # Policy heads - one for each intersection
        self.policy_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_size)
            ) for _ in range(num_intersections)
        ])
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Batch of states (B, state_size) or single state (state_size,)
            
        Returns:
            Tensor of shape (B, num_intersections, action_size) or 
            (num_intersections, action_size) for a single state
        """
        # Handle different input formats
        original_shape = x.shape
        is_single_state = (len(original_shape) == 1)
        
        if is_single_state:
            # Add batch dimension
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        
        # Reshape to (batch_size, num_intersections, features_per_intersection)
        x = x.view(batch_size, self.num_intersections, self.features_per_intersection)
        
        # Process each intersection's features
        intersection_features = []
        
        for i in range(self.num_intersections):
            # Extract features for this intersection
            features = x[:, i, :]
            
            # Apply shared layers
            if batch_size > 1:
                h = F.relu(self.shared_bn1(self.shared_fc1(features)))
            else:
                h = F.relu(self.shared_fc1(features))
            
            # Apply policy head
            q_values = self.policy_heads[i](h)
            intersection_features.append(q_values)
        
        # Combine all intersection outputs
        output = torch.stack(intersection_features, dim=1)
        
        # Remove batch dimension for single state
        if is_single_state:
            output = output.squeeze(0)
            
        return output 