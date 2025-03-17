"""
PPO Agent
=========
Implementation of Proximal Policy Optimization for traffic light control.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import logging
from .base import BaseAgent

logger = logging.getLogger("TrafficRL.Agents.PPO")

class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""
    def __init__(self, state_size, action_size, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        # Shared feature layers
        self.features = nn.Sequential(
            nn.Linear(state_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy network)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_size),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value network)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        features = self.features(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value

class PPOAgent(BaseAgent):
    """PPO agent implementation."""
    
    def __init__(self, state_size, action_size, config=None):
        """
        Initialize the PPO agent.
        
        Args:
            state_size: Dimensionality of the state space
            action_size: Dimensionality of the action space
            config: Configuration dictionary
        """
        super(PPOAgent, self).__init__(state_size, action_size)
        
        # Default configuration
        self.config = {
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_ratio": 0.2,
            "value_coef": 1.0,  # Value loss coefficient
            "entropy_coef": 0.01,  # Entropy coefficient
            "batch_size": 64,
            "update_epochs": 10,
            "hidden_dim": 256,
            "device": "auto",
            "max_grad_norm": 0.5
        }
        
        # Update config with provided values
        if config:
            self.config.update(config)
        
        # Set device
        if self.config.get("device", "auto") == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.get("device", "cpu"))
        
        # Initialize networks
        self.policy = ActorCritic(state_size, action_size, self.config.get("hidden_dim", 256)).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config.get("learning_rate", 3e-4))
        
        # Initialize memory
        self.reset()
        
        logger.info(f"PPOAgent initialized with state_size={state_size}, action_size={action_size}, device={self.device}")
    
    def act(self, state, eval_mode=False):
        """
        Choose an action based on the current state.
        
        Args:
            state: Current state
            eval_mode: If True, use evaluation (greedy) policy
        
        Returns:
            Selected action
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs, _ = self.policy(state)
            
            if eval_mode:
                action = torch.argmax(action_probs).item()
            else:
                dist = Categorical(action_probs)
                action = dist.sample().item()
                self.log_probs.append(dist.log_prob(torch.tensor(action)))
            
            return action
    
    def step(self, state, action, reward, next_state, done):
        """
        Process a step of experience.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        
        if done:
            self._process_episode()
    
    def learn(self, experiences=None):
        """
        Update the agent's policy using PPO.
        
        Args:
            experiences: Optional batch of experiences (not used in PPO)
        """
        if len(self.states) < self.config["batch_size"]:
            return
        
        # Convert stored data to tensors
        states = torch.FloatTensor(self.states).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.stack(self.log_probs).detach()
        advantages = torch.FloatTensor(self.advantages).to(self.device)
        returns = torch.FloatTensor(self.returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.config["update_epochs"]):
            # Generate random permutation for mini-batches
            indices = torch.randperm(len(states))
            
            for start_idx in range(0, len(states), self.config["batch_size"]):
                idx = indices[start_idx:start_idx + self.config["batch_size"]]
                
                # Get current policy outputs
                action_probs, state_values = self.policy(states[idx])
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(actions[idx])
                
                # Calculate ratio
                ratio = torch.exp(new_log_probs - old_log_probs[idx])
                
                # Calculate surrogate losses
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1.0 - self.config["clip_ratio"],
                                  1.0 + self.config["clip_ratio"]) * advantages[idx]
                
                # Calculate policy loss
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss
                value_loss = nn.MSELoss()(state_values.squeeze(), returns[idx])
                
                # Calculate entropy loss
                entropy_loss = -dist.entropy().mean()
                
                # Total loss
                loss = (policy_loss + 
                       self.config["value_coef"] * value_loss + 
                       self.config["entropy_coef"] * entropy_loss)
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config["max_grad_norm"])
                self.optimizer.step()
        
        # Clear memory after update
        self.reset()
    
    def _process_episode(self):
        """Process the collected episode data."""
        rewards = np.array(self.rewards)
        next_states = torch.FloatTensor(self.next_states).to(self.device)
        
        # Calculate returns and advantages
        with torch.no_grad():
            _, next_values = self.policy(next_states)
            next_values = next_values.squeeze().cpu().numpy()
        
        # Calculate returns
        returns = []
        running_return = 0
        for r, v, done in zip(reversed(rewards), reversed(next_values), reversed(self.dones)):
            if done:
                running_return = 0
            running_return = r + self.config["gamma"] * running_return
            returns.insert(0, running_return)
        
        # Calculate advantages using GAE
        advantages = []
        gae = 0
        for r, v, next_v, done in zip(reversed(rewards), reversed(next_values), 
                                    reversed(next_values[1:] + [0]), reversed(self.dones)):
            if done:
                gae = 0
            delta = r + self.config["gamma"] * next_v - v
            gae = delta + self.config["gamma"] * self.config["gae_lambda"] * gae
            advantages.insert(0, gae)
        
        self.returns = returns
        self.advantages = advantages
    
    def save(self, filepath):
        """
        Save the agent's model to a file.
        
        Args:
            filepath: Path to save the model
            
        Returns:
            Whether the save was successful
        """
        try:
            torch.save({
                'policy_state_dict': self.policy.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config
            }, filepath)
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load(self, filepath):
        """
        Load the agent's model from a file.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Whether the load was successful
        """
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # Handle different checkpoint structures
            if isinstance(checkpoint, dict):
                if 'policy_state_dict' in checkpoint:
                    self.policy.load_state_dict(checkpoint['policy_state_dict'])
                    logger.info("Loaded policy state from checkpoint")
                else:
                    # Try to find any state_dict in the checkpoint
                    state_dict_keys = [k for k in checkpoint.keys() if 'state_dict' in k]
                    if state_dict_keys:
                        logger.info(f"Found alternative state dict key: {state_dict_keys[0]}")
                        self.policy.load_state_dict(checkpoint[state_dict_keys[0]], strict=False)
                    else:
                        logger.warning("No state dict found in checkpoint, trying direct load")
                        # Try to load the entire checkpoint as a state dict
                        self.policy.load_state_dict(checkpoint, strict=False)
                
                # Load optimizer if it exists
                if 'optimizer_state_dict' in checkpoint:
                    try:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        logger.info("Loaded optimizer state")
                    except Exception as e:
                        logger.warning(f"Error loading optimizer state: {e}")
                
                # Update config with saved config if it exists
                if 'config' in checkpoint:
                    # Only update non-critical parameters
                    saved_config = checkpoint['config']
                    for key, value in saved_config.items():
                        if key not in ['state_size', 'action_size', 'device']:
                            self.config[key] = value
                    logger.info("Updated config from checkpoint")
            
            elif isinstance(checkpoint, torch.nn.Module):
                # If checkpoint is a model, try to load it directly
                logger.info("Checkpoint appears to be a direct model, loading state dict")
                self.policy.load_state_dict(checkpoint.state_dict(), strict=False)
            
            else:
                # Try direct loading as a state dict
                logger.info("Attempting to load as direct state dict")
                self.policy.load_state_dict(checkpoint, strict=False)
            
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def reset(self):
        """Reset the agent's memory."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.returns = []
        self.advantages = [] 