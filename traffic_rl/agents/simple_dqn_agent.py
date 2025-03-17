"""
Simple DQN Agent
==============
A basic implementation of DQN agent with simplified features.
"""

import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import logging

# Import models
from traffic_rl.models.simple_dqn import SimpleDQN
from traffic_rl.memory.replay_buffer import ReplayBuffer

logger = logging.getLogger("TrafficRL.Agent")

class SimpleDQNAgent:
    """
    Simple DQN Agent for traffic light control.
    Basic implementation with experience replay and target network.
    """
    def __init__(self, state_size, action_size, config):
        """Initialize the agent."""
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        
        # Get device - auto-detect if set to 'auto' or if missing
        if config.get("device", "auto") == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.get("device", "cpu"))
        logger.info(f"Using device: {self.device}")
        
        # Q-Networks
        self.local_network = SimpleDQN(state_size, action_size, 
                                     hidden_dim=config.get("hidden_dim", 128)).to(self.device)
        self.target_network = SimpleDQN(state_size, action_size, 
                                       hidden_dim=config.get("hidden_dim", 128)).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.local_network.parameters(), 
                                    lr=config.get("learning_rate", 0.001))
        
        # Replay buffer
        self.memory = ReplayBuffer(config.get("buffer_size", 10000), 
                                 config.get("batch_size", 64))
        
        # Epsilon for exploration
        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_end = config.get("epsilon_end", 0.01)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        
        # Initialize time step
        self.t_step = 0
        
        # Training metrics
        self.loss_history = []
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and learn if it's time."""
        try:
            # Convert to numpy arrays with consistent shapes
            state_np = np.array(state, dtype=np.float32)
            action_np = np.array([[action]], dtype=np.int64)
            reward_np = np.array([[reward]], dtype=np.float32)
            next_state_np = np.array(next_state, dtype=np.float32)
            done_np = np.array([[done]], dtype=np.float32)
            
            # Save experience in replay memory
            self.memory.add(state_np, action_np, reward_np, next_state_np, done_np)
            
            # Increment the time step
            self.t_step += 1
            
            # Learn every UPDATE_EVERY time steps if enough samples
            if len(self.memory) > self.config["batch_size"]:
                if self.t_step % self.config["target_update"] == 0:
                    experiences = self.memory.sample()
                    self.learn(experiences)
        except Exception as e:
            logger.error(f"Error in step() method: {e}")
    
    def act(self, state, eval_mode=False):
        """Choose an action based on the current state."""
        try:
            # Convert state to tensor
            if isinstance(state, torch.Tensor):
                state_tensor = state.float()
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Move to device
            state_tensor = state_tensor.to(self.device)
            
            # Set to evaluation mode
            self.local_network.eval()
            
            with torch.no_grad():
                action_values = self.local_network(state_tensor)
            
            # Set back to training mode
            self.local_network.train()
            
            # Epsilon-greedy action selection
            if not eval_mode and random.random() < self.epsilon:
                return int(random.randrange(self.action_size))
            else:
                return int(np.argmax(action_values.cpu().data.numpy()))
                
        except Exception as e:
            logger.error(f"Error in act() method: {e}")
            return int(random.randrange(self.action_size))
    
    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples."""
        if experiences is None:
            return
            
        try:
            states, actions, rewards, next_states, dones = experiences
            
            # Move to device
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)
            
            # Get max Q values from target network
            with torch.no_grad():
                Q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
            
            # Compute Q targets for current states
            Q_targets = rewards + (self.config["gamma"] * Q_targets_next * (1 - dones))
            
            # Get expected Q values from local model
            Q_expected = self.local_network(states).gather(1, actions)
            
            # Compute loss
            loss = F.mse_loss(Q_expected, Q_targets)
            
            # Store loss for monitoring
            self.loss_history.append(loss.item())
            
            # Minimize the loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update target network
            if self.t_step % self.config["target_update"] == 0:
                self.target_network.load_state_dict(self.local_network.state_dict())
            
            # Update epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        except Exception as e:
            logger.error(f"Error in learn() method: {e}")
    
    def save(self, filename):
        """Save the model."""
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            torch.save({
                'local_network_state_dict': self.local_network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'loss_history': self.loss_history
            }, filename)
            logger.info(f"Model saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def get_q_values(self, state):
        """Get Q-values for a given state."""
        try:
            if isinstance(state, torch.Tensor):
                state_tensor = state.float()
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            state_tensor = state_tensor.to(self.device)
            
            self.local_network.eval()
            with torch.no_grad():
                q_values = self.local_network(state_tensor)
            self.local_network.train()
            
            return q_values.cpu().data.numpy()[0]
                
        except Exception as e:
            logger.error(f"Error in get_q_values() method: {e}")
            return np.zeros(self.action_size)
    
    def load(self, filename):
        """
        Load the agent's model and training state from a file.
        
        Args:
            filename: Path to the saved model file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(filename):
                logger.error(f"Model file {filename} does not exist")
                return False
                
            # Load checkpoint
            checkpoint = torch.load(filename)
            
            # Load network state dictionaries - check for different key names
            if 'local_network_state_dict' in checkpoint:
                self.local_network.load_state_dict(checkpoint['local_network_state_dict'])
                if 'target_network_state_dict' in checkpoint:
                    self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
                else:
                    # If no target network, copy from local
                    self.target_network.load_state_dict(checkpoint['local_network_state_dict'])
            elif 'model_state_dict' in checkpoint:
                self.local_network.load_state_dict(checkpoint['model_state_dict'])
                if 'target_network_state_dict' in checkpoint:
                    self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
                else:
                    # If no target network, copy from local
                    self.target_network.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Try to load the network directly (no state_dict key)
                logger.info("Attempting to load model with direct state dict")
                self.local_network.load_state_dict(checkpoint)
                self.target_network.load_state_dict(checkpoint)
            
            # Load optimizer state if it exists
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load epsilon but ensure it's within bounds
            if 'epsilon' in checkpoint:
                self.epsilon = max(
                    min(
                        checkpoint['epsilon'],
                        self.config.get("epsilon_start", 1.0)
                    ),
                    self.config.get("epsilon_end", 0.01)
                )
            
            logger.info(f"Model loaded successfully from {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model components: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False 