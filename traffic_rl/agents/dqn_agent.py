"""
DQN Agent
========
Agent for Deep Q-Network reinforcement learning.
"""

import os
import numpy as np
import torch
import torch.optim as optim
import random
import logging

# Import models
from traffic_rl.models.dqn import DQN, IntersectionDQN
from traffic_rl.models.dueling_dqn import DuelingDQN

# Import memory buffers
from traffic_rl.memory.replay_buffer import ReplayBuffer
from traffic_rl.memory.prioritized_buffer import PrioritizedReplayBuffer

logger = logging.getLogger("TrafficRL.Agent")

class DQNAgent:
    """
    DQN Agent for traffic light control.
    
    This agent implements Deep Q-Learning with experience replay and target network.
    With the IntersectionDQN model, it can control each intersection independently.
    """
    def __init__(self, state_size, action_size, config):
        """Initialize the agent."""
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        
        # Calculate number of intersections and features per intersection
        self.num_intersections = config.get("grid_size", 4) ** 2
        self.features_per_intersection = state_size // self.num_intersections
        logger.info(f"State size: {state_size}, Intersections: {self.num_intersections}, Features per intersection: {self.features_per_intersection}")
        
        # Flag for independent intersection control
        self.independent_control = config.get("independent_control", True)
        
        # Get device - auto-detect if set to 'auto'
        if config["device"] == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config["device"])
        logger.info(f"Using device: {self.device}")
        
        # Q-Networks - select based on config
        if self.independent_control:
            # Use IntersectionDQN for independent control of each intersection
            logger.info("Using independent intersection control with IntersectionDQN")
            self.local_network = IntersectionDQN(
                features_per_intersection=self.features_per_intersection, 
                output_dim=action_size,
                hidden_dim=config.get("hidden_dim", 256)
            ).to(self.device)
            self.target_network = IntersectionDQN(
                features_per_intersection=self.features_per_intersection, 
                output_dim=action_size,
                hidden_dim=config.get("hidden_dim", 256)
            ).to(self.device)
        elif config.get("advanced_options", {}).get("dueling_network", False):
            logger.info("Using Dueling DQN architecture")
            self.local_network = DuelingDQN(state_size, action_size, hidden_dim=config.get("hidden_dim", 256)).to(self.device)
            self.target_network = DuelingDQN(state_size, action_size, hidden_dim=config.get("hidden_dim", 256)).to(self.device)
        else:
            logger.info("Using standard DQN architecture")
            self.local_network = DQN(state_size, action_size, hidden_dim=config.get("hidden_dim", 256)).to(self.device)
            self.target_network = DQN(state_size, action_size, hidden_dim=config.get("hidden_dim", 256)).to(self.device)
        
        # Optimizer with learning rate
        self.optimizer = optim.Adam(
            self.local_network.parameters(), 
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 0)  # L2 regularization
        )
        
        # Learning rate scheduler for stability
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=config.get("lr_step_size", 100),
            gamma=config.get("lr_decay", 0.5)
        )
        
        # Replay buffer - standard or prioritized
        if config.get("advanced_options", {}).get("prioritized_replay", False):
            logger.info("Using Prioritized Experience Replay")
            self.memory = PrioritizedReplayBuffer(
                config["buffer_size"], 
                config["batch_size"],
                alpha=config.get("per_alpha", 0.6),
                beta=config.get("per_beta", 0.4)
            )
            self.use_prioritized = True
        else:
            logger.info("Using standard Experience Replay")
            self.memory = ReplayBuffer(config["buffer_size"], config["batch_size"])
            self.use_prioritized = False
        
        # Epsilon for exploration
        self.epsilon = config["epsilon_start"]
        self.epsilon_end = config["epsilon_end"]
        self.epsilon_decay = config["epsilon_decay"]
        
        # Gradient clipping value
        self.grad_clip = config.get("grad_clip", 1.0)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
        # Training metrics
        self.loss_history = []
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and learn if it's time."""
        try:
            # Convert action to proper format based on independent control mode
            if self.independent_control:
                # With independent control, action should be an array of actions
                if not isinstance(action, (list, np.ndarray)):
                    logger.warning(f"Expected array of actions, got {type(action)}. Converting to array.")
                    action = np.array([action])
                    
                # Convert each action to scalar if needed
                action_np = np.array(action, dtype=np.int64).reshape(1, -1)  # Shape [1, num_intersections]
            else:
                # Handle single global action
                if isinstance(action, (np.ndarray, list, torch.Tensor)):
                    if hasattr(action, 'item'):
                        action = action.item()  # For PyTorch tensors
                    elif isinstance(action, np.ndarray) and action.size == 1:
                        action = action.item()  # For NumPy arrays
                    elif len(action) == 1:
                        action = action[0]  # For lists
                
                action_np = np.array([[action]], dtype=np.int64)  # Shape [1, 1]
            
            # Convert to numpy arrays with consistent shapes
            state_np = np.array(state, dtype=np.float32)
            reward_np = np.array([[reward]], dtype=np.float32)  # Shape [1, 1]
            next_state_np = np.array(next_state, dtype=np.float32)
            done_np = np.array([[done]], dtype=np.float32)  # Shape [1, 1]
            
            # Save experience in replay memory
            self.memory.add(state_np, action_np, reward_np, next_state_np, done_np)
            
            # Increment the time step
            self.t_step += 1
            
            # Check if enough samples are available in memory
            if len(self.memory) > self.config["batch_size"]:
                # If enough samples, learn every UPDATE_EVERY time steps
                if self.t_step % self.config["target_update"] == 0:
                    experiences = self.memory.sample()
                    self.learn(experiences)
        except Exception as e:
            logger.error(f"Error in step() method: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Continue without breaking training
    
    def act(self, state, eval_mode=False):
        """
        Choose an action based on the current state.
        
        Args:
            state: Current state
            eval_mode: If True, greedy policy is used
        
        Returns:
            Selected action or array of actions (one per intersection)
        """
        try:
            # Handle different input types for state
            if isinstance(state, torch.Tensor):
                # Already a tensor
                state_tensor = state.float()
                if state_tensor.dim() == 1:
                    state_tensor = state_tensor.unsqueeze(0)
            else:
                # Convert to numpy array first, ensuring correct dtype
                try:
                    # Try direct conversion if already numpy-like
                    np_state = np.array(state, dtype=np.float32)
                    state_tensor = torch.tensor(np_state, dtype=torch.float32).unsqueeze(0)
                except Exception as e:
                    logger.warning(f"Error converting state to tensor: {e}")
                    # Fallback method
                    state_tensor = torch.FloatTensor([state]).unsqueeze(0)
            
            # Move to device
            state_tensor = state_tensor.to(self.device)
            
            # Set to evaluation mode
            self.local_network.eval()
            
            with torch.no_grad():
                action_values = self.local_network(state_tensor)
            
            # Set back to training mode
            self.local_network.train()
            
            if self.independent_control:
                # Handle independent control - return array of actions
                actions = []
                for i in range(self.num_intersections):
                    if not eval_mode and random.random() < self.epsilon:
                        # Random action for this intersection
                        actions.append(random.randrange(self.action_size))
                    else:
                        # Greedy action for this intersection
                        if len(action_values.shape) == 3:  # [batch, num_intersections, action_size]
                            intersection_values = action_values[0, i]
                        else:  # [num_intersections, action_size]
                            intersection_values = action_values[i]
                        actions.append(int(torch.argmax(intersection_values).cpu().item()))
                
                return actions
            else:
                # Standard global action selection
                if not eval_mode and random.random() < self.epsilon:
                    return int(random.randrange(self.action_size))
                else:
                    # Make sure to return a plain Python int, not a numpy or torch type
                    return int(np.argmax(action_values.cpu().data.numpy()))
                
        except Exception as e:
            logger.error(f"Error in act() method: {e}")
            # Return random action as fallback
            if self.independent_control:
                return [int(random.randrange(self.action_size)) for _ in range(self.num_intersections)]
            else:
                return int(random.randrange(self.action_size))
    
    def learn(self, experiences):
        """
        Update value parameters using given batch of experience tuples.
        
        Args:
            experiences: Tuple of (s, a, r, s', done) tuples and possibly weights and indices
        """
        # Check if experiences is None (could happen if there was an error in sampling)
        if experiences is None:
            logger.warning("No valid experiences to learn from")
            return
            
        try:
            # Handle different formats for prioritized vs standard replay
            if self.use_prioritized:
                states, actions, rewards, next_states, dones, weights, indices = experiences
            else:
                states, actions, rewards, next_states, dones = experiences
                weights = torch.ones_like(rewards)  # Uniform weights
            
            # Move to device
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)
            weights = weights.to(self.device)
            
            if self.independent_control:
                # For independent intersection control, reshape actions if needed
                if actions.dim() == 2 and actions.size(1) != self.num_intersections:
                    logger.warning(f"Actions shape mismatch: {actions.shape}, expected second dim to be {self.num_intersections}")
                    # Try to adapt actions - might need adjustment based on actual data
                    if actions.size(1) == 1:
                        # Broadcast single action to all intersections
                        actions = actions.expand(-1, self.num_intersections)
                
                # Get predictions for each intersection
                q_values = self.local_network(states)  # shape: [batch, num_intersections, action_size]
                
                # Handle next state predictions based on double DQN flag
                if self.config.get("advanced_options", {}).get("double_dqn", False):
                    with torch.no_grad():
                        # Get action selections from online network for each intersection
                        next_actions = torch.argmax(self.local_network(next_states), dim=2)  # [batch, num_intersections]
                        next_actions = next_actions.unsqueeze(2)  # [batch, num_intersections, 1]
                        
                        # Get target Q values for selected actions
                        target_q_values = self.target_network(next_states)  # [batch, num_intersections, action_size]
                        Q_targets_next = torch.gather(target_q_values, 2, next_actions)  # [batch, num_intersections, 1]
                else:
                    # Standard DQN: Use max Q value from target network for each intersection
                    with torch.no_grad():
                        target_q_values = self.target_network(next_states)  # [batch, num_intersections, action_size]
                        Q_targets_next = torch.max(target_q_values, dim=2, keepdim=True)[0]  # [batch, num_intersections, 1]
                
                # Reshape rewards and dones to match Q_targets_next dimensions
                # Q_targets_next shape is [batch, num_intersections, 1]
                batch_size = rewards.size(0)
                
                # Check the shapes before reshaping
                logger.debug(f"Rewards shape: {rewards.shape}, Dones shape: {dones.shape}, Q_targets_next shape: {Q_targets_next.shape}")
                
                # Handle rewards and dones with different approaches based on their shape
                if rewards.dim() == 2:
                    if rewards.size(1) == 1:
                        # If rewards is [batch, 1], create a new tensor with the right shape
                        rewards_reshaped = rewards.expand(-1, 1).reshape(batch_size, 1, 1).expand(-1, self.num_intersections, -1)
                    else:
                        # If rewards has another size in dim 1, reshape it completely
                        # First flatten it to [batch*dim1] then reshape to [batch, num_intersections, 1]
                        # This approach works if batch*dim1 is divisible by batch*num_intersections
                        rewards_reshaped = rewards.reshape(batch_size * rewards.size(1), 1)
                        target_size = batch_size * self.num_intersections
                        
                        # If sizes don't match, use a subset or duplicate as needed
                        if rewards_reshaped.size(0) > target_size:
                            rewards_reshaped = rewards_reshaped[:target_size]
                        elif rewards_reshaped.size(0) < target_size:
                            # Duplicate the elements to match the target size
                            repeat_factor = (target_size + rewards_reshaped.size(0) - 1) // rewards_reshaped.size(0)
                            rewards_reshaped = rewards_reshaped.repeat(repeat_factor, 1)[:target_size]
                        
                        rewards_reshaped = rewards_reshaped.reshape(batch_size, self.num_intersections, 1)
                else:
                    # If rewards has unexpected dimensions, create a simple tensor of ones
                    rewards_reshaped = torch.ones_like(Q_targets_next) * rewards.mean()
                
                # Similarly for dones
                if dones.dim() == 2:
                    if dones.size(1) == 1:
                        # If dones is [batch, 1], create a new tensor with the right shape
                        dones_reshaped = dones.expand(-1, 1).reshape(batch_size, 1, 1).expand(-1, self.num_intersections, -1)
                    else:
                        # If dones has another size in dim 1, reshape it completely
                        dones_reshaped = dones.reshape(batch_size * dones.size(1), 1)
                        target_size = batch_size * self.num_intersections
                        
                        # If sizes don't match, use a subset or duplicate as needed
                        if dones_reshaped.size(0) > target_size:
                            dones_reshaped = dones_reshaped[:target_size]
                        elif dones_reshaped.size(0) < target_size:
                            # Duplicate the elements to match the target size
                            repeat_factor = (target_size + dones_reshaped.size(0) - 1) // dones_reshaped.size(0)
                            dones_reshaped = dones_reshaped.repeat(repeat_factor, 1)[:target_size]
                        
                        dones_reshaped = dones_reshaped.reshape(batch_size, self.num_intersections, 1)
                else:
                    # If dones has unexpected dimensions, create a simple tensor of zeros
                    dones_reshaped = torch.zeros_like(Q_targets_next)
                
                # Compute Q targets for current states - apply to all intersections
                Q_targets = rewards_reshaped + (self.config["gamma"] * Q_targets_next * (1 - dones_reshaped))
                
                # Get expected Q values for selected actions
                # Reshape actions to gather properly if needed
                if actions.dim() == 2:  # [batch, num_intersections]
                    actions = actions.unsqueeze(2)  # [batch, num_intersections, 1]
                
                Q_expected = torch.gather(q_values, 2, actions)  # [batch, num_intersections, 1]
                
                # Reshape weights to match Q_targets dimensions if needed
                if weights.dim() == 2 and weights.size(1) == 1:
                    weights = weights.reshape(batch_size, 1, 1).expand(-1, self.num_intersections, -1)
                elif weights.dim() != 3 or weights.size(1) != self.num_intersections:
                    weights = torch.ones_like(Q_targets)
                
                # Compute loss with importance sampling weights
                td_errors = Q_targets - Q_expected
                loss = (weights * td_errors.pow(2)).mean()
                
            else:
                # Standard DQN learning for global control
                # Ensure actions tensor has the right shape for gather operation
                if actions.dim() != 2 or actions.size(1) != 1:
                    logger.warning(f"Reshaping actions tensor from {actions.shape}")
                    # If actions is [batch], reshape to [batch, 1]
                    if actions.dim() == 1:
                        actions = actions.unsqueeze(1)
                    # If actions is [batch, action_dim, 1], reshape to [batch, 1]
                    elif actions.dim() == 3:
                        actions = actions.squeeze(2)
                
                # Double DQN: Use online network to select action, target network to evaluate
                if self.config.get("advanced_options", {}).get("double_dqn", False):
                    with torch.no_grad():
                        # Get action selection from online network
                        next_actions = self.local_network(next_states).detach().max(1)[1].unsqueeze(1)
                        # Get Q values from target network for selected actions
                        Q_targets_next = self.target_network(next_states).gather(1, next_actions)
                else:
                    # Standard DQN: Use max Q value from target network
                    with torch.no_grad():
                        Q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
                
                # Compute Q targets for current states
                Q_targets = rewards + (self.config["gamma"] * Q_targets_next * (1 - dones))
                
                # Get Q values from local model for all actions
                q_values = self.local_network(states)
                
                # Get expected Q values for selected actions
                Q_expected = q_values.gather(1, actions)
                
                # Compute loss with importance sampling weights for prioritized replay
                td_errors = Q_targets - Q_expected
                loss = (weights * td_errors.pow(2)).mean()
            
            # Store loss for monitoring
            self.loss_history.append(loss.item())
            
            # Zero gradients, perform a backward pass, and update the weights
            self.optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.local_network.parameters(), self.grad_clip)
            
            # Update weights
            self.optimizer.step()
            
            # Update target network
            self._soft_update()
            
            # Update prioritized replay buffer if using it
            if self.use_prioritized:
                with torch.no_grad():
                    # Use abs TD errors as priorities
                    priorities = td_errors.abs().detach().cpu().numpy() + 1e-6  # Add small value to avoid zero priorities
                    if self.independent_control:
                        # Average TD errors across intersections
                        priorities = priorities.mean(axis=1).squeeze()
                    else:
                        priorities = priorities.squeeze()
                    
                    self.memory.update_priorities(indices, priorities)
            
            # Update epsilon value for exploration
            self._update_epsilon()
            
        except Exception as e:
            logger.error(f"Error in learn() method: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _soft_update(self, tau=0.001):
        """Soft update target network parameters."""
        for target_param, local_param in zip(self.target_network.parameters(), self.local_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def _update_epsilon(self):
        """Update epsilon for exploration."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        """Save model weights and configuration."""
        try:
            save_dict = {
                'model_state': self.local_network.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'state_size': self.state_size,
                'action_size': self.action_size,
                'features_per_intersection': self.features_per_intersection,
                'num_intersections': self.num_intersections,
                'independent_control': self.independent_control
            }
            torch.save(save_dict, filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load(self, filepath):
        """Load model weights."""
        try:
            # Set weights_only=False to handle PyTorch 2.6 compatibility
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            
            # Check if the saved model has the correct architecture
            if 'independent_control' in checkpoint and checkpoint['independent_control'] != self.independent_control:
                logger.warning(f"Loading model with different control mode: saved={checkpoint['independent_control']}, current={self.independent_control}")
            
            self.local_network.load_state_dict(checkpoint['model_state'])
            self.target_network.load_state_dict(checkpoint['model_state'])
            
            if 'optimizer_state' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            
            if 'epsilon' in checkpoint:
                self.epsilon = checkpoint['epsilon']
                
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
