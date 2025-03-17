"""
Entity-Aware DQN Model
====================
Neural network model for car entity-based Deep Q-Learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger("TrafficRL.Models")

class CarEntityEncoder(nn.Module):
    """
    Encodes information about individual cars at each intersection.
    This module processes features of cars at an intersection to create
    a fixed-size representation regardless of the number of cars.
    """
    def __init__(self, car_feature_dim=4, hidden_dim=64, output_dim=128):
        super(CarEntityEncoder, self).__init__()
        
        # Car feature processing
        self.car_encoder = nn.Sequential(
            nn.Linear(car_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Attention mechanism for cars
        self.attention = nn.Linear(hidden_dim, 1)
        
        # Output layer to get fixed representation
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, car_features, car_mask=None):
        """
        Process a variable number of cars at an intersection.
        
        Args:
            car_features: Tensor of shape [batch_size, max_cars, car_feature_dim]
            car_mask: Boolean mask of shape [batch_size, max_cars] where True indicates a valid car
                     (needed to handle variable numbers of cars)
        
        Returns:
            Fixed-size representation of all cars at the intersection
        """
        batch_size, max_cars, _ = car_features.shape
        
        # Default mask if none provided (assume all cars are valid)
        if car_mask is None:
            car_mask = torch.ones(batch_size, max_cars, dtype=torch.bool, device=car_features.device)
        
        # Process each car's features
        # Reshape to process all cars in the batch together
        car_features_flat = car_features.view(-1, car_features.size(-1))
        encoded_cars_flat = self.car_encoder(car_features_flat)
        encoded_cars = encoded_cars_flat.view(batch_size, max_cars, -1)
        
        # Apply attention to focus on important cars
        attention_scores = self.attention(encoded_cars)  # [batch_size, max_cars, 1]
        
        # Apply mask to attention scores (set masked positions to -inf)
        if car_mask is not None:
            mask_expanded = (~car_mask).unsqueeze(-1)  # [batch_size, max_cars, 1]
            attention_scores = attention_scores.masked_fill(mask_expanded, float('-inf'))
        
        # Softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, max_cars, 1]
        
        # Apply attention weights to encoded cars
        weighted_encoding = encoded_cars * attention_weights
        
        # Sum over cars dimension to get fixed-size representation
        pooled_encoding = weighted_encoding.sum(dim=1)  # [batch_size, hidden_dim]
        
        # Final output transformation
        output = self.output_layer(pooled_encoding)  # [batch_size, output_dim]
        
        return output


class EntityDQN(nn.Module):
    """
    Deep Q-Network for traffic light control with car entity awareness.
    Processes both intersection level features and individual car information.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, car_feature_dim=4, max_cars_per_intersection=10):
        super(EntityDQN, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_cars_per_intersection = max_cars_per_intersection
        
        # Car entity encoder (shared across all intersections)
        self.car_encoder = CarEntityEncoder(
            car_feature_dim=car_feature_dim,
            hidden_dim=hidden_dim // 2,
            output_dim=hidden_dim // 2
        )
        
        # Intersection feature encoder
        self.intersection_encoder = nn.Sequential(
            nn.Linear(5, hidden_dim // 2),  # 5 basic features per intersection
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
        )
        
        # Combined processing after concatenating car and intersection features
        self.combined_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
        )
        
        # Output layer for Q-values
        self.q_value_head = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, state, car_features=None, car_masks=None):
        """
        Forward pass through the network.
        
        Args:
            state: Tensor of shape [batch_size, num_intersections, 5] containing basic intersection features 
                  (NS_cars, EW_cars, light_state, NS_waiting, EW_waiting)
            car_features: Optional tensor of shape [batch_size, num_intersections, max_cars, car_feature_dim] 
                        containing features of individual cars at each intersection
            car_masks: Optional tensor of shape [batch_size, num_intersections, max_cars] 
                      containing masks for valid cars
        
        Returns:
            Q-values for each action at each intersection
        """
        batch_size = state.size(0)
        num_intersections = state.size(1) if state.dim() > 1 else 1
        
        # Ensure state has correct shape
        if state.dim() == 2:  # [batch_size, flattened_features]
            state = state.view(batch_size, num_intersections, -1)
            
        # Process intersection features
        # Reshape to process all intersections together
        intersection_features = state.view(-1, state.size(-1))  # [batch_size*num_intersections, 5]
        encoded_intersections = self.intersection_encoder(intersection_features)
        
        # If car features are provided, process them
        if car_features is not None:
            # Reshape to process all intersections together
            # Original shape: [batch_size, num_intersections, max_cars, car_feature_dim]
            # Reshape to: [batch_size*num_intersections, max_cars, car_feature_dim]
            car_features_reshaped = car_features.view(
                -1, self.max_cars_per_intersection, car_features.size(-1)
            )
            
            # Reshape masks if provided
            if car_masks is not None:
                car_masks_reshaped = car_masks.view(-1, self.max_cars_per_intersection)
            else:
                car_masks_reshaped = None
                
            # Process car features
            encoded_cars = self.car_encoder(car_features_reshaped, car_masks_reshaped)
            
            # Concatenate intersection and car features
            combined_features = torch.cat([encoded_intersections, encoded_cars], dim=1)
        else:
            # If no car features, use a zero tensor as placeholder
            car_encoding_size = encoded_intersections.size(1)
            zero_encoding = torch.zeros(
                encoded_intersections.size(0), 
                car_encoding_size, 
                device=encoded_intersections.device
            )
            combined_features = torch.cat([encoded_intersections, zero_encoding], dim=1)
            
        # Process combined features
        processed_features = self.combined_network(combined_features)
        
        # Calculate Q-values
        q_values = self.q_value_head(processed_features)
        
        # Reshape back to [batch_size, num_intersections, action_dim]
        q_values = q_values.view(batch_size, num_intersections, -1)
        
        # If we only have one intersection, squeeze that dimension
        if num_intersections == 1:
            q_values = q_values.squeeze(1)
            
        return q_values
