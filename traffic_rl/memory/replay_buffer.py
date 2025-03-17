"""
Experience Replay Buffer
======================
Replay memory for storing and sampling experiences.
"""

import numpy as np
import torch
import random
from collections import namedtuple, deque
import logging

logger = logging.getLogger("TrafficRL.Memory")

class ReplayBuffer:
    """
    Experience replay buffer to store and sample transitions.
    """
    def __init__(self, buffer_size, batch_size):
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", 
                                     field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to the buffer."""
        experience = self.experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self):
        """Sample a batch of experiences from the buffer."""
        try:
            experiences = random.sample(self.buffer, k=min(self.batch_size, len(self.buffer)))
            
            # Ensure all elements are properly processed
            valid_experiences = []
            for e in experiences:
                if e is not None:
                    valid_experiences.append(e)
            
            if not valid_experiences:
                # Return None if no valid experiences
                return None
            
            # Convert to numpy arrays first to handle different shapes
            states_np = np.array([e.state for e in valid_experiences])
            
            # Ensure actions are reshaped properly for gather operation
            actions_np = np.array([e.action for e in valid_experiences])
            
            # Handle actions with different shapes (for independent intersection control)
            if actions_np.ndim == 3:
                # Only squeeze if the dimension has size 1
                if actions_np.shape[2] == 1:
                    actions_np = actions_np.squeeze(2)
                # Otherwise, keep the shape for independent actions per intersection
            elif actions_np.ndim == 1:
                actions_np = np.expand_dims(actions_np, 1)
                
            rewards_np = np.array([e.reward for e in valid_experiences])
            if rewards_np.ndim == 3:
                if rewards_np.shape[2] == 1:  # Only squeeze if dim size is 1
                    rewards_np = rewards_np.squeeze(2)
            elif rewards_np.ndim == 1:
                rewards_np = np.expand_dims(rewards_np, 1)
                
            next_states_np = np.array([e.next_state for e in valid_experiences])
            
            dones_np = np.array([e.done for e in valid_experiences])
            if dones_np.ndim == 3:
                if dones_np.shape[2] == 1:  # Only squeeze if dim size is 1
                    dones_np = dones_np.squeeze(2)
            elif dones_np.ndim == 1:
                dones_np = np.expand_dims(dones_np, 1)
            
            # Convert to torch tensors
            states = torch.tensor(states_np, dtype=torch.float32)
            actions = torch.tensor(actions_np, dtype=torch.long)
            rewards = torch.tensor(rewards_np, dtype=torch.float32)
            next_states = torch.tensor(next_states_np, dtype=torch.float32)
            dones = torch.tensor(dones_np, dtype=torch.float32)
            
            return (states, actions, rewards, next_states, dones)
        
        except Exception as e:
            logger.error(f"Error sampling from replay buffer: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)