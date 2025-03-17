"""
Entity DQN Agent
===============
Agent that uses the entity-aware DQN model for reinforcement learning.
"""

import numpy as np
import torch
import random
import logging
from collections import defaultdict

from traffic_rl.agents.dqn_agent import DQNAgent
from traffic_rl.models.entity_dqn import EntityDQN
from traffic_rl.models.dueling_dqn import DuelingDQN
from traffic_rl.environment.car import Direction, CarState

logger = logging.getLogger("TrafficRL.Agent")

class EntityDQNAgent(DQNAgent):
    """
    DQN Agent that works with individual car entities.
    
    This agent extends the base DQNAgent but uses an entity-aware
    neural network that can process both intersection-level features
    and individual car features.
    """
    def __init__(self, state_size, action_size, config):
        """
        Initialize the entity-aware DQN agent.
        
        Args:
            state_size: Size of the intersection state (observation space)
            action_size: Size of the action space
            config: Configuration dictionary
        """
        # Skip DQNAgent's __init__ and call the grandparent's __init__
        # We'll set up our networks differently
        self.state_size = state_size
        self.action_size = action_size
        self.config = config if config else {}
        
        # Car-specific parameters
        self.car_feature_dim = config.get("car_feature_dim", 4)
        self.max_cars_per_intersection = config.get("max_cars_per_intersection", 10)
        self.grid_size = config.get("grid_size", 4)
        self.num_intersections = self.grid_size * self.grid_size
        
        # Get device - auto-detect if set to 'auto' or if missing
        if self.config.get("device", "auto") == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.get("device", "cpu"))
        logger.info(f"Using device: {self.device}")
        
        # Initialize the entity-aware networks
        hidden_dim = self.config.get("hidden_dim", 256)
        
        # Q-Networks - select based on config
        if self.config.get("advanced_options", {}).get("dueling_network", False):
            logger.info("Using Dueling DQN architecture (Note: not fully implemented for entity mode)")
            # For now, fall back to regular EntityDQN until we implement a Dueling version
            self.local_network = EntityDQN(
                state_dim=5,  # 5 features per intersection
                action_dim=action_size,
                hidden_dim=hidden_dim,
                car_feature_dim=self.car_feature_dim,
                max_cars_per_intersection=self.max_cars_per_intersection
            ).to(self.device)
            
            self.target_network = EntityDQN(
                state_dim=5,
                action_dim=action_size,
                hidden_dim=hidden_dim,
                car_feature_dim=self.car_feature_dim,
                max_cars_per_intersection=self.max_cars_per_intersection
            ).to(self.device)
        else:
            logger.info("Using entity-aware DQN architecture")
            self.local_network = EntityDQN(
                state_dim=5,
                action_dim=action_size,
                hidden_dim=hidden_dim,
                car_feature_dim=self.car_feature_dim,
                max_cars_per_intersection=self.max_cars_per_intersection
            ).to(self.device)
            
            self.target_network = EntityDQN(
                state_dim=5,
                action_dim=action_size,
                hidden_dim=hidden_dim,
                car_feature_dim=self.car_feature_dim,
                max_cars_per_intersection=self.max_cars_per_intersection
            ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.local_network.parameters(),
            lr=self.config.get("learning_rate", 0.001),
            weight_decay=self.config.get("weight_decay", 0)
        )
        
        # Initialize memory buffer (reusing DQNAgent's memory initialization)
        if self.config.get("advanced_options", {}).get("prioritized_replay", False):
            logger.info("Using prioritized experience replay")
            from traffic_rl.memory.prioritized_buffer import PrioritizedReplayBuffer
            self.memory = PrioritizedReplayBuffer(
                self.config.get("buffer_size", 10000),
                self.config.get("batch_size", 64),
                self.config.get("advanced_options", {}).get("per_alpha", 0.6)
            )
            self.priority_beta = self.config.get("advanced_options", {}).get("per_beta", 0.4)
        else:
            logger.info("Using standard experience replay")
            from traffic_rl.memory.replay_buffer import ReplayBuffer
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
    
    def extract_car_features(self, env):
        """
        Extract features for individual cars at each intersection.
        
        Args:
            env: Traffic simulation environment with active_cars
            
        Returns:
            car_features: Tensor of car features for each intersection
            car_mask: Boolean mask indicating which car slots contain valid cars
        """
        # Initialize array for car features for each intersection
        # Shape: [num_intersections, max_cars_per_intersection, car_feature_dim]
        car_features = np.zeros((
            self.num_intersections, 
            self.max_cars_per_intersection, 
            self.car_feature_dim
        ), dtype=np.float32)
        
        # Initialize mask to track which car slots are filled
        # Shape: [num_intersections, max_cars_per_intersection]
        car_mask = np.zeros((self.num_intersections, self.max_cars_per_intersection), dtype=bool)
        
        # Group cars by the nearest intersection
        intersection_cars = defaultdict(list)
        
        for car in env.active_cars:
            # Get car position and find nearest intersection
            x, y = car.position
            grid_x, grid_y = int(min(max(0, x), self.grid_size-1)), int(min(max(0, y), self.grid_size-1))
            intersection_idx = grid_y * self.grid_size + grid_x
            
            # Only add car if we haven't reached max cars for this intersection
            if len(intersection_cars[intersection_idx]) < self.max_cars_per_intersection:
                intersection_cars[intersection_idx].append(car)
        
        # Process cars at each intersection
        for intersection_idx, cars in intersection_cars.items():
            for i, car in enumerate(cars):
                if i >= self.max_cars_per_intersection:
                    break  # Safety check
                    
                # Extract features for this car
                # 1. Is car waiting (0 or 1)
                # 2. Car's direction (one-hot: N, E, S, W)
                # 3. Car's position relative to intersection center (x, y)
                # 4. Car's speed
                # 5. Car's waiting time
                
                # Mark this slot as containing a valid car
                car_mask[intersection_idx, i] = True
                
                # Extract basic features
                is_waiting = float(car.state == CarState.WAITING)
                speed = min(car.speed / car.max_speed, 1.0)  # Normalize to [0, 1]
                waiting_time = min(car.waiting_time / 10.0, 1.0)  # Normalize to [0, 1]
                
                # Extract direction (one-hot encoding)
                # 0=North, 1=East, 2=South, 3=West
                direction_onehot = np.zeros(4, dtype=np.float32)
                if car.direction == Direction.NORTH:
                    direction_onehot[0] = 1.0
                elif car.direction == Direction.EAST:
                    direction_onehot[1] = 1.0
                elif car.direction == Direction.SOUTH:
                    direction_onehot[2] = 1.0
                elif car.direction == Direction.WEST:
                    direction_onehot[3] = 1.0
                
                # Compute car position relative to intersection center
                # Normalize to [-1, 1] range
                int_x, int_y = grid_x + 0.5, grid_y + 0.5  # Center of intersection
                rel_x = (car.position[0] - int_x) / 0.5  # Normalize to [-1, 1]
                rel_y = (car.position[1] - int_y) / 0.5  # Normalize to [-1, 1]
                rel_pos = np.clip([rel_x, rel_y], -1, 1)
                
                # Combine all features
                # Format: [is_waiting, direction_onehot, rel_position, speed, waiting_time]
                car_feature = np.concatenate([
                    [is_waiting], 
                    direction_onehot,
                    rel_pos,
                    [speed],
                    [waiting_time]
                ])
                
                # Store features (take only the first car_feature_dim features)
                car_features[intersection_idx, i, :] = car_feature[:self.car_feature_dim]
        
        return car_features, car_mask
    
    def act(self, state, env=None, eval_mode=False):
        """
        Choose an action based on the current state and environment.
        
        Args:
            state: Current state (intersection features)
            env: Traffic simulation environment (for extracting car features)
            eval_mode: If True, greedy policy is used
            
        Returns:
            Selected action
        """
        try:
            # Convert state to tensor
            if isinstance(state, torch.Tensor):
                state_tensor = state.float()
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32)
            
            # Reshape to [batch_size, num_intersections, features]
            if state_tensor.dim() == 1:
                # Single flattened state
                num_features = 5  # Basic features per intersection
                num_intersections = state_tensor.size(0) // num_features
                state_tensor = state_tensor.view(1, num_intersections, num_features)
            elif state_tensor.dim() == 2 and state_tensor.size(0) == 1:
                # Batch of 1 flattened state
                num_features = 5  # Basic features per intersection
                num_intersections = state_tensor.size(1) // num_features
                state_tensor = state_tensor.view(1, num_intersections, num_features)
            
            # Add batch dimension if missing
            if state_tensor.dim() == 2:
                state_tensor = state_tensor.unsqueeze(0)
            
            # Move to device
            state_tensor = state_tensor.to(self.device)
            
            # Extract car features if environment is provided
            if env is not None:
                car_features, car_mask = self.extract_car_features(env)
                car_features_tensor = torch.tensor(car_features, dtype=torch.float32).unsqueeze(0).to(self.device)
                car_mask_tensor = torch.tensor(car_mask, dtype=torch.bool).unsqueeze(0).to(self.device)
            else:
                car_features_tensor = None
                car_mask_tensor = None
            
            # Set to evaluation mode
            self.local_network.eval()
            
            with torch.no_grad():
                action_values = self.local_network(
                    state_tensor,
                    car_features_tensor,
                    car_mask_tensor
                )
            
            # Set back to training mode
            self.local_network.train()
            
            # If multiple intersections, we get an action for each one
            # For now, just return the same action for all intersections (simplification)
            action_values = action_values.cpu().data.numpy()
            if action_values.ndim > 1:
                action_values = action_values.mean(axis=1)  # Average across intersections
            
            # Epsilon-greedy action selection
            if not eval_mode and random.random() < self.epsilon:
                return int(random.randrange(self.action_size))
            else:
                return int(np.argmax(action_values))
                
        except Exception as e:
            logger.error(f"Error in act() method: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return random action as fallback
            return int(random.randrange(self.action_size))
    
    def step(self, state, action, reward, next_state, done, env=None):
        """
        Save experience in replay memory, and learn if it's time.
        
        Args:
            state: Current state
            action: Selected action
            reward: Received reward
            next_state: Next state
            done: Whether the episode is done
            env: Traffic simulation environment (for extracting car features)
        """
        try:
            # For now, we still use the standard replay buffer without car features
            # In a full implementation, we would also store car features in the buffer
            
            # Add experience to memory
            self.memory.add(state, action, reward, next_state, done)
            
            # Learn every UPDATE_EVERY time steps
            self.t_step = (self.t_step + 1) % self.update_every
            if self.t_step == 0:
                # If enough samples are available in memory, get a random sample and learn
                if len(self.memory) > self.memory.batch_size:
                    experiences = self.memory.sample()
                    self.learn(experiences)
            
            # Update epsilon value
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
        except Exception as e:
            logger.error(f"Error in step() method: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def learn(self, experiences):
        """
        Update value parameters using given batch of experience tuples.
        
        Args:
            experiences: Tuple of (s, a, r, s', done) tuples and possibly weights and indices
        """
        try:
            # Extract components from experiences tuple
            if isinstance(experiences, tuple) and len(experiences) > 5:
                states, actions, rewards, next_states, dones, weights, indices = experiences
                prioritized = True
            else:
                states, actions, rewards, next_states, dones = experiences
                prioritized = False
                
            # Get device
            device = next(self.local_network.parameters()).device
            
            # Convert to tensors and move to the right device
            # Note: we don't have car features here as they're not stored in memory buffer
            # This is a simplification - in a full implementation, we'd store them
            states_tensor = torch.FloatTensor(states).to(device)
            actions_tensor = torch.LongTensor(actions).to(device)
            rewards_tensor = torch.FloatTensor(rewards).to(device).view(-1, 1)
            next_states_tensor = torch.FloatTensor(next_states).to(device)
            dones_tensor = torch.FloatTensor(dones).to(device).view(-1, 1)
            
            # For entity DQN, we need to reshape state for both networks
            # Create dummy car features with zeros
            batch_size = states.shape[0]
            num_intersections = self.num_intersections
            car_feature_dim = self.car_feature_dim
            max_cars = self.max_cars_per_intersection
            
            # Dummy car features and masks (all zeros)
            dummy_car_features = torch.zeros(
                batch_size, num_intersections, max_cars, car_feature_dim, 
                device=device
            )
            dummy_car_masks = torch.zeros(
                batch_size, num_intersections, max_cars, 
                device=device,
                dtype=torch.bool  # Explicitly use boolean type
            )
            
            # Reshape state tensors
            states_tensor = states_tensor.view(batch_size, num_intersections, -1)
            next_states_tensor = next_states_tensor.view(batch_size, num_intersections, -1)
            
            # Get predicted Q-values for current states
            predicted_q_values = self.local_network(
                states_tensor, 
                car_features=dummy_car_features,
                car_masks=dummy_car_masks
            )
            
            # Reshape if needed
            if predicted_q_values.dim() > 2:
                # If we have multiple intersections, take the mean
                predicted_q_values = predicted_q_values.mean(dim=1)
            
            # Get Q-values for chosen actions
            predicted_q_values = predicted_q_values.gather(1, actions_tensor)
            
            # Compute target Q-values
            with torch.no_grad():
                # Get target Q-values for next states
                target_q_values = self.target_network(
                    next_states_tensor,
                    car_features=dummy_car_features,
                    car_masks=dummy_car_masks
                )
                
                # Reshape if needed
                if target_q_values.dim() > 2:
                    # If we have multiple intersections, take the mean
                    target_q_values = target_q_values.mean(dim=1)
                
                # Get max Q-values for next states
                next_max_q_values = target_q_values.max(1)[0].unsqueeze(1)
                
                # Compute target Q-values
                target_q_values = rewards_tensor + (self.gamma * next_max_q_values * (1 - dones_tensor))
            
            # Compute loss
            loss = torch.nn.MSELoss()(predicted_q_values, target_q_values)
            
            # If using prioritized replay, update priorities
            if prioritized:
                # Calculate TD error for prioritized replay
                with torch.no_grad():
                    td_errors = torch.abs(target_q_values - predicted_q_values).cpu().numpy()
                
                # Update priorities in the replay buffer
                for i, td_error in enumerate(td_errors):
                    self.memory.update_priorities(indices[i], td_error[0])
                    
                # Apply importance sampling weights
                weight_tensor = torch.FloatTensor(weights).to(device).view(-1, 1)
                loss = (loss * weight_tensor).mean()
            
            # Zero gradients, perform gradient step, and update target network
            self.optimizer.zero_grad()
            loss.backward()
            
            # Apply gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.local_network.parameters(), self.grad_clip)
                
            self.optimizer.step()
            
            # Update target network
            self.soft_update()
            
            # Add loss to metrics
            self.losses.append(loss.item())
            
            # Store Q-values for monitoring
            with torch.no_grad():
                self.q_values.append(predicted_q_values.mean().item())
                
        except Exception as e:
            logger.error(f"Error in learn() method: {e}")
            import traceback
            logger.error(traceback.format_exc())
