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
from traffic_rl.models.dqn import DQN
from traffic_rl.models.dueling_dqn import DuelingDQN

# Import memory buffers
from traffic_rl.memory.replay_buffer import ReplayBuffer
from traffic_rl.memory.prioritized_buffer import PrioritizedReplayBuffer

logger = logging.getLogger("TrafficRL.Agent")

class DQNAgent:
    """
    DQN Agent for traffic light control.
    
    This agent implements Deep Q-Learning with experience replay and target network.
    """
    def __init__(self, state_size, action_size, config):
        """Initialize the agent."""
        self.state_size = state_size
        self.action_size = action_size
        self.config = config if config else {}
        
        # Get device - handle "mps" specifically for Mac
        if self.config.get("device", "auto") == "auto":
            # Auto-detect available device
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("MPS (Metal Performance Shaders) device detected and will be used")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("CUDA device detected and will be used")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU device")
        elif self.config.get("device") == "mps":
            # Explicitly check if MPS is available
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("Using MPS (Metal Performance Shaders) for Mac acceleration")
            else:
                logger.warning("MPS requested but not available, falling back to CPU")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.config.get("device", "cpu"))
        
        logger.info(f"Using device: {self.device}")
        
        # Determine if we should use per-intersection model
        self.use_per_intersection = self.config.get("use_per_intersection", False)
        self.num_intersections = self.config.get("num_intersections", 16)  # Default for 4x4 grid
        
        # Import appropriate models
        if self.use_per_intersection:
            try:
                from traffic_rl.models.per_intersection_dqn import PerIntersectionDQN
                
                logger.info("Using Per-Intersection DQN architecture")
                self.local_network = PerIntersectionDQN(
                    state_size, 
                    action_size, 
                    self.num_intersections, 
                    hidden_dim=self.config.get("hidden_dim", 64)
                ).to(self.device)
                
                self.target_network = PerIntersectionDQN(
                    state_size, 
                    action_size, 
                    self.num_intersections, 
                    hidden_dim=self.config.get("hidden_dim", 64)
                ).to(self.device)
            except ImportError:
                logger.warning("Per-Intersection model not found, falling back to standard DQN")
                self.use_per_intersection = False
                
        # Use standard models if per-intersection is not enabled
        if not self.use_per_intersection:
            # Q-Networks - select based on config
            if self.config.get("advanced_options", {}).get("dueling_network", False):
                from traffic_rl.models.dueling_dqn import DuelingDQN
                logger.info("Using Dueling DQN architecture")
                self.local_network = DuelingDQN(state_size, action_size, hidden_dim=self.config.get("hidden_dim", 256)).to(self.device)
                self.target_network = DuelingDQN(state_size, action_size, hidden_dim=self.config.get("hidden_dim", 256)).to(self.device)
            else:
                from traffic_rl.models.dqn import DQN
                logger.info("Using standard DQN architecture")
                self.local_network = DQN(state_size, action_size, hidden_dim=self.config.get("hidden_dim", 256)).to(self.device)
                self.target_network = DQN(state_size, action_size, hidden_dim=self.config.get("hidden_dim", 256)).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.local_network.parameters(),
            lr=self.config.get("learning_rate", 0.001),
            weight_decay=self.config.get("weight_decay", 0)
        )
        
        # Initialize replay buffer
        if self.config.get("advanced_options", {}).get("prioritized_replay", False):
            logger.info("Using prioritized experience replay")
            self.memory = PrioritizedReplayBuffer(
                self.config.get("buffer_size", 10000),
                self.config.get("batch_size", 64),
                self.config.get("alpha", 0.6)
            )
            self.priority_beta = self.config.get("beta", 0.4)
        else:
            logger.info("Using standard experience replay")
            self.memory = ReplayBuffer(
                self.config.get("buffer_size", 10000),
                self.config.get("batch_size", 64)
            )
        
        # Initialize training parameters
        self.gamma = self.config.get("gamma", 0.99)
        self.tau = self.config.get("tau", 1e-3)
        self.update_every = self.config.get("update_every", 4)
        
        # Initialize exploration parameters
        self.epsilon = self.config.get("epsilon_start", 1.0)
        self.epsilon_end = self.config.get("epsilon_end", 0.01)
        self.epsilon_decay = self.config.get("epsilon_decay", 0.995)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
        # Initialize additional metrics
        self.q_values = []
        
        # Gradient clipping value
        self.grad_clip = self.config.get("grad_clip", 1.0)
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and learn if it's time."""
        try:
            # Handle action arrays (per-intersection actions)
            # For replay buffer, we'll store the global action (same for all intersections)
            # or the first action if they're different
            if isinstance(action, (np.ndarray, list)):
                # Check if all actions are the same
                if hasattr(action, 'size') and action.size > 1:
                    # Use the most frequent action as the "global" action
                    unique_actions, counts = np.unique(action, return_counts=True)
                    global_action = unique_actions[np.argmax(counts)]
                elif len(action) > 0:
                    # Just use the first action
                    global_action = action[0]
                else:
                    global_action = 0  # Default action
            elif isinstance(action, torch.Tensor):
                if action.numel() > 1:
                    # Use the most frequent action
                    unique_actions, counts = torch.unique(action, return_counts=True)
                    global_action = unique_actions[torch.argmax(counts)].item()
                else:
                    global_action = action.item()
            else:
                global_action = action
            
            # Convert to numpy arrays with consistent shapes
            state_np = np.array(state, dtype=np.float32)
            action_np = np.array([[global_action]], dtype=np.int64)  # Store the global action
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
            Selected action(s) - one per intersection
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
            
            # Determine the number of intersections based on the state size and action size
            # Each intersection has 5 features in the observation space
            num_intersections = state_tensor.shape[1] // 5 if state_tensor.dim() > 1 else state_tensor.shape[0] // 5
            
            # Initialize action array
            actions = np.zeros(num_intersections, dtype=int)
            
            # Choose actions per intersection
            if not eval_mode and random.random() < self.epsilon:
                # Random action for each intersection
                for i in range(num_intersections):
                    actions[i] = int(random.randrange(self.action_size))
            else:
                # Reshape action values if needed (for a simple network that outputs all actions at once)
                # If the network outputs a single value, it's a global action (apply to all intersections)
                if len(action_values.shape) == 1 or (len(action_values.shape) == 2 and action_values.shape[1] == self.action_size):
                    # Global action - apply the same action to all intersections
                    best_action = int(np.argmax(action_values.cpu().data.numpy()))
                    actions.fill(best_action)
                else:
                    # Per-intersection actions
                    action_values_np = action_values.cpu().data.numpy()
                    for i in range(num_intersections):
                        actions[i] = int(np.argmax(action_values_np[i]))
            
            return actions
                
        except Exception as e:
            logger.error(f"Error selecting action: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return a random action as fallback
            return np.random.randint(0, self.action_size, size=num_intersections)
    
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
            if self.config.get("advanced_options", {}).get("prioritized_replay", False):
                states, actions, rewards, next_states, dones, weights, indices = experiences
            else:
                states, actions, rewards, next_states, dones = experiences
                weights = torch.ones_like(rewards)  # Uniform weights
            
            # Ensure actions tensor has the right shape for gather operation
            if actions.dim() != 2 or actions.size(1) != 1:
                logger.warning(f"Reshaping actions tensor from {actions.shape}")
                # If actions is [batch], reshape to [batch, 1]
                if actions.dim() == 1:
                    actions = actions.unsqueeze(1)
                # If actions is [batch, action_dim, 1], reshape to [batch, 1]
                elif actions.dim() == 3:
                    actions = actions.squeeze(2)
            
            # Move to device
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)
            weights = weights.to(self.device)
            
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
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
            
            # Get Q values from local model for all actions
            q_values = self.local_network(states)
            
            # Get expected Q values for selected actions
            Q_expected = q_values.gather(1, actions)
            
            # Compute loss with importance sampling weights for prioritized replay
            td_errors = Q_targets - Q_expected
            loss = (weights * td_errors.pow(2)).mean()
            
            # Store loss for monitoring
            self.q_values.append(loss.item())
            
            # Minimize the loss
            self.optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.local_network.parameters(), self.grad_clip)
                
            self.optimizer.step()
            
            # Update target network
            if self.t_step % self.update_every == 0:
                self._update_target_network()
            
            # Update epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        except Exception as e:
            logger.error(f"Error in learn() method: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def save(self, filename):
        """Save the model."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            torch.save({
                'local_network_state_dict': self.local_network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'q_values': self.q_values
            }, filename)
            logger.info(f"Model saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def get_q_values(self, state):
        """
        Get Q-values for a given state.
        
        Args:
            state: Current state
            
        Returns:
            Q-values for all actions
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
                q_values = self.local_network(state_tensor)
            
            # Set back to training mode
            self.local_network.train()
            
            # Return as numpy array
            return q_values.cpu().data.numpy()[0]
                
        except Exception as e:
            logger.error(f"Error in get_q_values() method: {e}")
            # Return zeros as fallback
            return np.zeros(self.action_size)
    
    def load(self, filename):
        """Load the model."""
        if not os.path.isfile(filename):
            logger.warning(f"Model file {filename} not found")
            return False
        
        try:
            # Try to load with regular torch.load
            checkpoint = torch.load(filename)
        except Exception as e:
            logger.warning(f"Failed to load model with regular torch.load: {e}")
            try:
                # Try loading with CPU map_location for models saved on different devices
                checkpoint = torch.load(filename, map_location=torch.device('cpu'))
            except Exception as e2:
                logger.error(f"Failed to load model: {e2}")
                return False
        
        try:
            # First, check what type of checkpoint this is
            if isinstance(checkpoint, dict):
                # If checkpoint is a dictionary, look for state dicts
                # Try loading the local network state dict if it exists
                if 'local_network_state_dict' in checkpoint:
                    try:
                        self.local_network.load_state_dict(checkpoint['local_network_state_dict'], strict=False)
                        logger.info("Loaded local network from local_network_state_dict")
                    except Exception as e:
                        logger.warning(f"Error loading local network: {e}")
                        
                        # Try to create a new model that matches the saved architecture
                        try:
                            old_state_dict = checkpoint['local_network_state_dict']
                            input_size = old_state_dict['fc1.weight'].shape[1]
                            output_size = old_state_dict['fc3.weight'].shape[0]
                            hidden_size = old_state_dict['fc1.weight'].shape[0]
                            
                            logger.info(f"Detected architecture: input={input_size}, hidden={hidden_size}, output={output_size}")
                            
                            # Create a new model with matching architecture
                            from traffic_rl.models.dqn import DQN
                            self.local_network = DQN(input_size, output_size, hidden_dim=hidden_size).to(self.device)
                            self.target_network = DQN(input_size, output_size, hidden_dim=hidden_size).to(self.device)
                            
                            # Try loading again
                            self.local_network.load_state_dict(checkpoint['local_network_state_dict'])
                            logger.info("Successfully loaded model with adjusted architecture")
                        except Exception as e2:
                            logger.error(f"Failed to load with adjusted architecture: {e2}")
                            return False
                
                # Try loading the target network state dict if it exists
                if 'target_network_state_dict' in checkpoint:
                    try:
                        self.target_network.load_state_dict(checkpoint['target_network_state_dict'], strict=False)
                        logger.info("Loaded target network from target_network_state_dict")
                    except Exception as e:
                        logger.warning(f"Error loading target network: {e}")
                        # If we successfully loaded local network, copy it to target
                        self.target_network.load_state_dict(self.local_network.state_dict())
                # Else copy local to target if target wasn't loaded
                elif hasattr(self, 'local_network'):
                    self.target_network.load_state_dict(self.local_network.state_dict())
                
                # Load optimizer if it exists
                if 'optimizer_state_dict' in checkpoint:
                    try:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        logger.info("Loaded optimizer state")
                    except Exception as e:
                        logger.warning(f"Error loading optimizer state: {e}")
                
                # Load epsilon
                if 'epsilon' in checkpoint:
                    self.epsilon = max(
                        min(checkpoint['epsilon'], self.config.get("epsilon_start", 1.0)),
                        self.config.get("epsilon_end", 0.01)
                    )
                    logger.info(f"Loaded epsilon: {self.epsilon}")
                
                # Load history data if available
                if 'q_values' in checkpoint:
                    self.q_values = checkpoint['q_values']
                    logger.info("Loaded q-values history")
                    
            elif isinstance(checkpoint, torch.nn.Module):
                # If checkpoint is a model, try to load it directly
                logger.info("Checkpoint appears to be a direct model, loading state dict")
                state_dict = checkpoint.state_dict()
                self.local_network.load_state_dict(state_dict, strict=False)
                self.target_network.load_state_dict(state_dict, strict=False)
            else:
                # Try loading as a direct state dict
                logger.info("Attempting to load as direct state dict")
                self.local_network.load_state_dict(checkpoint, strict=False)
                self.target_network.load_state_dict(checkpoint, strict=False)
            
            logger.info(f"Model loaded successfully from {filename}")
            return True
        except Exception as e:
            logger.error(f"Error loading model components: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _update_target_network(self):
        """Soft update model parameters: θ_target = τ*θ_local + (1 - τ)*θ_target"""
        for target_param, local_param in zip(self.target_network.parameters(), self.local_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        logger.debug("Target network updated")
