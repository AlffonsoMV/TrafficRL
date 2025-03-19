"""
Traffic Simulation Environment
============================
A custom Gym environment for traffic simulation.
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import logging

logger = logging.getLogger("TrafficRL.Environment")

class TrafficSimulation(gym.Env):
    """
    Custom Gym environment for traffic simulation.
    
    Represents a grid of intersections controlled by traffic lights.
    Each intersection has four incoming lanes (North, East, South, West).
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 180}
    
    def __init__(self, config, visualization=False, random_seed=None, traffic_pattern=None):
        """
        Initialize the traffic simulation environment.
        
        Args:
            config: Configuration dictionary
            visualization: Whether to enable visualization
            random_seed: Random seed for reproducibility
            traffic_pattern: Traffic pattern to use (uniform, rush_hour, weekend)
        """
        super(TrafficSimulation, self).__init__()
        
        # Store configuration
        self.config = config
        
        # Environment configuration
        self.grid_size = config.get("grid_size", 4)
        self.max_cars = config.get("max_cars", 30)
        self.visualization = visualization
        
        # Set random seed if provided
        if random_seed is not None:
            self.np_random = np.random.RandomState(random_seed)
        else:
            self.np_random = np.random
        
        # Number of intersections in the grid
        self.num_intersections = self.grid_size * self.grid_size
        
        # Traffic light states: 0=North-South Green, 1=East-West Green
        self.action_space = spaces.Discrete(2)
        
        # Observation space: traffic density and light state for each intersection
        # For each intersection: [NS_density, EW_density, light_state, NS_waiting, EW_waiting]
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(self.num_intersections, 5),
            dtype=np.float32
        )
        
        # Initialize traffic pattern
        self.traffic_pattern = traffic_pattern or config.get("traffic_pattern", "uniform")
        
        # Make sure the pattern is valid
        if self.traffic_pattern not in config.get("traffic_patterns", {}):
            logger.warning(f"Unknown traffic pattern: {self.traffic_pattern}, defaulting to uniform")
            self.traffic_pattern = "uniform"
        
        self.traffic_config = config["traffic_patterns"][self.traffic_pattern]
        
        # Initialize visualization if enabled
        if self.visualization:
            try:
                self._init_visualization()
            except Exception as e:
                logger.warning(f"Could not initialize visualization: {e}")
                self.visualization = False
                
        # Reset the environment
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        # Set seed if provided
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        
        # Initialize traffic densities (random initial state)
        self.traffic_density = self.np_random.uniform(
            0.0, 0.5, size=(self.num_intersections, 2)
        )
        
        # Initialize traffic light states (all start with NS green)
        self.light_states = np.zeros(self.num_intersections, dtype=int)
        
        # Track waiting time for cars at each intersection
        self.waiting_time = np.zeros((self.num_intersections, 2))
        self.prev_waiting_time = np.zeros((self.num_intersections, 2))
        
        # Track number of cars passed through each intersection
        self.cars_passed = np.zeros((self.num_intersections, 2))
        self.prev_cars_passed = np.zeros((self.num_intersections, 2))
        
        # Track green light durations for each direction
        self.ns_green_duration = np.zeros(self.num_intersections)
        self.ew_green_duration = np.zeros(self.num_intersections)
        
        # Reset simulation time
        self.sim_time = 0
        self.step_count = 0
        
        # Generate observation
        observation = self._get_observation()
        
        # Info dictionary for the reset state
        info = {}
        
        # Return observation as required by newer Gymnasium API
        return observation, info
    
    def step(self, actions):
        """
        Take a step in the environment given the actions.
        
        Args:
            actions: Array of actions for each intersection (0=NS Green, 1=EW Green)
                    or a single action to apply to all intersections
        
        Returns:
            observation: Current observation
            reward: Reward from the action
            terminated: Whether the episode is done
            truncated: Whether the episode is truncated
            info: Additional information
        """
        try:
            # Increment step counter
            self.step_count += 1
            
            # Store previous state values for relative reward calculations
            self.prev_waiting_time = np.copy(self.waiting_time)
            self.prev_cars_passed = np.copy(self.cars_passed)
            
            # Handle both scalar and array inputs for actions
            if isinstance(actions, (int, np.integer, float, np.floating)):
                # If a single action is provided, convert to array
                actions_array = np.full(self.num_intersections, int(actions))
            elif isinstance(actions, (list, np.ndarray)):
                # If array-like with single value, convert to array of that value
                if len(actions) == 1:
                    actions_array = np.full(self.num_intersections, actions[0])
                elif len(actions) != self.num_intersections:
                    # If array with wrong length, broadcast or truncate
                    logger.warning(f"Actions array length {len(actions)} doesn't match num_intersections {self.num_intersections}")
                    actions_array = np.resize(actions, self.num_intersections)
                else:
                    # Correct length array
                    actions_array = np.array(actions)
            else:
                # Fallback for unexpected action type
                logger.warning(f"Unexpected action type: {type(actions)}, defaulting to all 0")
                actions_array = np.zeros(self.num_intersections, dtype=int)
            
            # Update traffic lights based on actions - direct control without timers
            for i in range(self.num_intersections):
                # Direct control: immediately set light state to agent's action
                self.light_states[i] = actions_array[i]
                    
            # Update duration trackers
            for i in range(self.num_intersections):
                if self.light_states[i] == 0:  # NS is green
                    self.ns_green_duration[i] += 1
                    self.ew_green_duration[i] = 0
                else:  # EW is green
                    self.ew_green_duration[i] += 1
                    self.ns_green_duration[i] = 0
                
            # Simulate traffic flow
            self._update_traffic()
            
            # Calculate reward
            reward = self._calculate_reward()
            
            # Generate observation
            observation = self._get_observation()
            
            # Update simulation time
            self.sim_time += 1
            
            # Check if episode is done
            terminated = False
            truncated = False
            
            # Additional info
            info = {
                'average_waiting_time': np.mean(self.waiting_time),
                'total_cars_passed': np.sum(self.cars_passed),
                'traffic_density': np.mean(self.traffic_density)
            }
            
            # Render if visualization is enabled
            if self.visualization:
                self.render()
            
            return observation, reward, terminated, truncated, info
            
        except Exception as e:
            logger.error(f"Error in environment step: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Return a safe fallback state
            fallback_obs = self._get_observation()
            return fallback_obs, 0.0, True, False, {"error": str(e)}
    
    def _update_traffic(self):
        """Simulate traffic flow and update densities with more realistic behavior."""
        try:
            # Track pre-update densities
            prev_density = np.copy(self.traffic_density)
            
            # Define speed factors based on density (congestion effects)
            # Higher density = slower traffic flow
            speed_factor_ns = 1.0 - 0.7 * prev_density[:, 0]  # Speed factor for NS direction
            speed_factor_ew = 1.0 - 0.7 * prev_density[:, 1]  # Speed factor for EW direction
            
            # Base flow rates
            base_flow_rate = 0.1  # Base flow rate with green light
            red_light_flow = 0.01  # Small flow even with red light (running red)
            
            for i in range(self.num_intersections):
                # Get current light state (0=NS Green, 1=EW Green)
                light = self.light_states[i]
                
                # Calculate flow rates with congestion effects
                if light == 0:  # NS green
                    # Green light flow rate affected by congestion
                    ns_flow_rate = base_flow_rate * speed_factor_ns[i]
                    # Small flow through red light (some cars run the red)
                    ew_flow_rate = red_light_flow * speed_factor_ew[i]
                else:  # EW green
                    # Small flow through red light
                    ns_flow_rate = red_light_flow * speed_factor_ns[i]
                    # Green light flow rate
                    ew_flow_rate = base_flow_rate * speed_factor_ew[i]
                
                # Calculate actual flow based on current density
                ns_cars_flow = min(self.traffic_density[i, 0], ns_flow_rate)
                ew_cars_flow = min(self.traffic_density[i, 1], ew_flow_rate)
                
                # Update densities and stats
                self.traffic_density[i, 0] -= ns_cars_flow
                self.traffic_density[i, 1] -= ew_cars_flow
                
                # Track cars that passed through
                self.cars_passed[i, 0] += ns_cars_flow * self.max_cars
                self.cars_passed[i, 1] += ew_cars_flow * self.max_cars
                
                # Calculate waiting time based on density and whether light is red
                if light == 0:  # NS Green
                    # Cars wait in EW direction
                    self.waiting_time[i, 1] += self.traffic_density[i, 1]
                else:  # EW Green
                    # Cars wait in NS direction
                    self.waiting_time[i, 0] += self.traffic_density[i, 0]
            
            # Simulate new cars arriving with daily patterns
            # Time of day effect (0=midnight, 0.5=noon, 1.0=midnight again)
            time_of_day = (self.sim_time % 1440) / 1440.0  # Normalize to [0,1]
            
            # Get traffic pattern configuration
            if self.traffic_pattern == "rush_hour":
                # Morning rush hour around 8am (time_of_day ~= 0.33)
                # Evening rush hour around 5pm (time_of_day ~= 0.71)
                morning_peak = self.traffic_config.get("morning_peak", 0.33)
                evening_peak = self.traffic_config.get("evening_peak", 0.71)
                peak_intensity = self.traffic_config.get("peak_intensity", 2.0)
                base_arrival = self.traffic_config.get("base_arrival", 0.03)
                
                rush_hour_factor = peak_intensity * (
                    np.exp(-20 * (time_of_day - morning_peak)**2) +  # Morning peak
                    np.exp(-20 * (time_of_day - evening_peak)**2)    # Evening peak
                )
            elif self.traffic_pattern == "weekend":
                # Weekend pattern: one peak around noon
                midday_peak = self.traffic_config.get("midday_peak", 0.5)
                peak_intensity = self.traffic_config.get("peak_intensity", 1.5)
                base_arrival = self.traffic_config.get("base_arrival", 0.02)
                
                rush_hour_factor = peak_intensity * np.exp(-10 * (time_of_day - midday_peak)**2)
            else:  # uniform pattern
                base_arrival = self.traffic_config.get("arrival_rate", 0.03)
                variability = self.traffic_config.get("variability", 0.01)
                rush_hour_factor = 0
            
            # Add randomness to arrival patterns
            for i in range(self.num_intersections):
                # New cars arrive from each direction with pattern effects
                arrival_factor = base_arrival * (1 + rush_hour_factor)
                ns_arrivals = arrival_factor * self.np_random.uniform(0.5, 1.5)
                ew_arrivals = arrival_factor * self.np_random.uniform(0.5, 1.5)
                
                # Add new cars (ensure density doesn't exceed 1.0)
                self.traffic_density[i, 0] = min(1.0, self.traffic_density[i, 0] + ns_arrivals)
                self.traffic_density[i, 1] = min(1.0, self.traffic_density[i, 1] + ew_arrivals)
            
            # Simulate traffic flow between adjacent intersections with directional flow
            if self.grid_size > 1:
                # Create a copy of current densities after individual intersection updates
                new_density = np.copy(self.traffic_density)
                
                # Calculate flow between intersections based on density gradients
                flow_between = 0.05  # Base rate of flow between intersections
                
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        idx = i * self.grid_size + j
                        
                        # For each direction, flow depends on density gradient
                        # Flow from high density to low density
                        
                        # North neighbor (i-1, j)
                        if i > 0:
                            north_idx = (i-1) * self.grid_size + j
                            # NS flow from current to north (if current has higher density)
                            density_diff = self.traffic_density[idx, 0] - self.traffic_density[north_idx, 0]
                            if density_diff > 0:
                                flow = flow_between * density_diff
                                actual_flow = min(flow, self.traffic_density[idx, 0] * 0.3)  # Limit to 30%
                                new_density[idx, 0] -= actual_flow
                                new_density[north_idx, 0] += actual_flow
                        
                        # South neighbor (i+1, j)
                        if i < self.grid_size - 1:
                            south_idx = (i+1) * self.grid_size + j
                            # NS flow from current to south
                            density_diff = self.traffic_density[idx, 0] - self.traffic_density[south_idx, 0]
                            if density_diff > 0:
                                flow = flow_between * density_diff
                                actual_flow = min(flow, self.traffic_density[idx, 0] * 0.3)
                                new_density[idx, 0] -= actual_flow
                                new_density[south_idx, 0] += actual_flow
                        
                        # West neighbor (i, j-1)
                        if j > 0:
                            west_idx = i * self.grid_size + (j-1)
                            # EW flow from current to west
                            density_diff = self.traffic_density[idx, 1] - self.traffic_density[west_idx, 1]
                            if density_diff > 0:
                                flow = flow_between * density_diff
                                actual_flow = min(flow, self.traffic_density[idx, 1] * 0.3)
                                new_density[idx, 1] -= actual_flow
                                new_density[west_idx, 1] += actual_flow
                        
                        # East neighbor (i, j+1)
                        if j < self.grid_size - 1:
                            east_idx = i * self.grid_size + (j+1)
                            # EW flow from current to east
                            density_diff = self.traffic_density[idx, 1] - self.traffic_density[east_idx, 1]
                            if density_diff > 0:
                                flow = flow_between * density_diff
                                actual_flow = min(flow, self.traffic_density[idx, 1] * 0.3)
                                new_density[idx, 1] -= actual_flow
                                new_density[east_idx, 1] += actual_flow
                
                # Update traffic density with new values, ensuring it stays in [0,1]
                self.traffic_density = np.clip(new_density, 0.0, 1.0)
                
        except Exception as e:
            logger.error(f"Error in traffic update: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _calculate_reward(self):
        """
        Calculate reward based on traffic flow efficiency with independent intersection control.
        
        This reward function:
        1. Calculates rewards PER INTERSECTION separately
        2. Focuses on local traffic conditions at each intersection
        3. Rewards IMPROVEMENTS rather than absolute values
        4. Provides clear feedback for good vs bad decisions
        
        Components for each intersection:
        1. Waiting time reduction (relative to previous step)
        2. Throughput increase (relative to previous step) 
        3. Good decision reward based on traffic direction imbalance
        4. Queue management penalty
        
        Returns:
            float: The combined reward across all intersections
        """
        try:
            # Initialize array to store rewards for each intersection
            intersection_rewards = np.zeros(self.num_intersections)
            
            # Calculate rewards for each intersection separately
            for i in range(self.num_intersections):
                # --- 1. Waiting time change (reward reductions, penalize increases) ---
                ns_waiting_change = self.waiting_time[i, 0] - self.prev_waiting_time[i, 0]
                ew_waiting_change = self.waiting_time[i, 1] - self.prev_waiting_time[i, 1]
                total_waiting_change = ns_waiting_change + ew_waiting_change
                
                # Negative change (reduction) is good, positive (increase) is bad
                # Scale based on magnitude - small improvements get small rewards
                waiting_reward = -total_waiting_change * 0.5
                
                # --- 2. Throughput change (reward increases) ---
                ns_throughput_change = self.cars_passed[i, 0] - self.prev_cars_passed[i, 0]
                ew_throughput_change = self.cars_passed[i, 1] - self.prev_cars_passed[i, 1]
                total_throughput_change = ns_throughput_change + ew_throughput_change
                
                # Positive change is good - scale to be meaningful but not dominant
                throughput_reward = total_throughput_change * 0.2
                
                # --- 3. Decision quality reward ---
                # This rewards making good decisions based on traffic state
                ns_density = self.traffic_density[i, 0]
                ew_density = self.traffic_density[i, 1]
                light_state = self.light_states[i]  # 0=NS green, 1=EW green
                
                # Measure traffic imbalance (difference between directions)
                imbalance = ns_density - ew_density
                
                # Perfect decision: green for higher density direction
                # Simplified binary reward for clear learning signal
                if (light_state == 0 and imbalance > 0.1) or (light_state == 1 and imbalance < -0.1):
                    # Good decision - clear density difference, correct light
                    decision_reward = 1.0
                elif abs(imbalance) <= 0.1:
                    # Neutral decision - densities are similar, either light is reasonable
                    decision_reward = 0.2
                else:
                    # Bad decision - giving green to less dense direction
                    decision_reward = -1.0
                
                # --- 4. Queue management penalty ---
                # Penalize high total queue length to avoid grid congestion
                total_density = ns_density + ew_density
                
                # Only penalize when queues are getting significant
                if total_density > 0.5:
                    queue_penalty = -(total_density - 0.5) * 0.5
                else:
                    queue_penalty = 0
                
                # Combine all components for this intersection
                # Balance the components to provide clear learning signal
                intersection_rewards[i] = waiting_reward + throughput_reward + decision_reward + queue_penalty
            
            # Log details of reward calculation periodically for debugging
            if self.step_count % 100 == 0:
                logger.debug(f"Intersection rewards: min={np.min(intersection_rewards):.2f}, "
                             f"max={np.max(intersection_rewards):.2f}, "
                             f"mean={np.mean(intersection_rewards):.2f}")
                
                # Log details for a sample intersection (first one)
                i = 0
                ns_density = self.traffic_density[i, 0]
                ew_density = self.traffic_density[i, 1]
                light_state = self.light_states[i]
                waiting_change = self.waiting_time[i, 0] + self.waiting_time[i, 1] - (self.prev_waiting_time[i, 0] + self.prev_waiting_time[i, 1])
                throughput_change = self.cars_passed[i, 0] + self.cars_passed[i, 1] - (self.prev_cars_passed[i, 0] + self.prev_cars_passed[i, 1])
                
                logger.debug(f"Sample intersection 0: NS={ns_density:.2f}, EW={ew_density:.2f}, "
                            f"light={'NS' if light_state==0 else 'EW'}, "
                            f"waiting_change={waiting_change:.2f}, throughput_change={throughput_change:.2f}, "
                            f"reward={intersection_rewards[0]:.2f}")
            
            # Return the combined reward (simply average across intersections)
            # This gives equal importance to all intersections regardless of grid size
            avg_reward = np.mean(intersection_rewards)
            
            # Apply a baseline so good performance is positive, bad performance is negative
            # This helps with interpreting the reward and for early stopping criteria
            final_reward = avg_reward + 1.0
            
            return final_reward
            
        except Exception as e:
            logger.error(f"Error calculating reward: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return -5.0  # Return a moderate negative reward on error
    
    def _get_observation(self):
        """
        Construct the observation from the current state.
        
        For each intersection, the observation includes:
        - NS traffic density (normalized)
        - EW traffic density (normalized)
        - Traffic light state (0 for NS green, 1 for EW green)
        - NS waiting time (normalized)
        - EW waiting time (normalized)
        """
        observation = np.zeros((self.num_intersections, 5), dtype=np.float32)
        
        for i in range(self.num_intersections):
            # Traffic density for NS and EW
            observation[i, 0] = self.traffic_density[i, 0]
            observation[i, 1] = self.traffic_density[i, 1]
            
            # Traffic light state
            observation[i, 2] = self.light_states[i]
            
            # Add waiting time information 
            observation[i, 3] = self.waiting_time[i, 0] / 10.0  # Normalized NS waiting
            observation[i, 4] = self.waiting_time[i, 1] / 10.0  # Normalized EW waiting
        
        return observation
    
    def _init_visualization(self):
        """Initialize pygame for visualization."""
        try:
            # Check if pygame is already initialized
            if pygame.get_init():
                logger.info("Pygame already initialized")
            else:
                pygame.init()
                
            self.screen_width = 800
            self.screen_height = 800
            
            # Try to set display mode with different fallback options
            try:
                # Try hardware accelerated mode first
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height),
                    pygame.HWSURFACE | pygame.DOUBLEBUF
                )
                logger.info("Using hardware accelerated rendering")
            except pygame.error:
                try:
                    # Fall back to software rendering
                    self.screen = pygame.display.set_mode(
                        (self.screen_width, self.screen_height)
                    )
                    logger.info("Using software rendering")
                except pygame.error:
                    # Last resort: dummy video driver for headless environments
                    os.environ['SDL_VIDEODRIVER'] = 'dummy'
                    self.screen = pygame.display.set_mode(
                        (self.screen_width, self.screen_height)
                    )
                    logger.warning("Using dummy video driver for headless rendering")
                
            pygame.display.set_caption("Traffic Simulation")
            self.clock = pygame.time.Clock()
            
            # Initialize font
            pygame.font.init()
            self.font = pygame.font.Font(None, 24)
            if not pygame.font.get_init():
                logger.warning("Failed to initialize pygame font module")
                
            logger.info("Visualization initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize visualization: {e}")
            self.visualization = False
            raise
    
    def render(self, mode='human'):
        """Render the environment."""
        if not self.visualization:
            return None
        
        try:
            # Check if pygame is still running
            if not pygame.get_init():
                logger.warning("Pygame not initialized, reinitializing...")
                self._init_visualization()
                
            # Fill background
            self.screen.fill((255, 255, 255))
            
            cell_width = self.screen_width // self.grid_size
            cell_height = self.screen_height // self.grid_size
            
            # Draw grid and traffic lights
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    idx = i * self.grid_size + j
                    
                    # Calculate position
                    x = j * cell_width
                    y = i * cell_height
                    
                    # Draw intersection
                    pygame.draw.rect(self.screen, (200, 200, 200), 
                                    (x, y, cell_width, cell_height))
                    
                    # Draw roads
                    road_width = min(cell_width, cell_height) // 4
                    
                    # NS road
                    pygame.draw.rect(self.screen, (100, 100, 100),
                                    (x + cell_width//2 - road_width//2, y, 
                                     road_width, cell_height))
                    
                    # EW road
                    pygame.draw.rect(self.screen, (100, 100, 100),
                                    (x, y + cell_height//2 - road_width//2, 
                                     cell_width, road_width))
                    
                    # Draw traffic light
                    light_radius = road_width // 2
                    light_x = x + cell_width // 2
                    light_y = y + cell_height // 2
                    
                    if self.light_states[idx] == 0:  # NS Green
                        # NS light green
                        pygame.draw.circle(self.screen, (0, 255, 0), 
                                          (light_x, light_y - light_radius), light_radius // 2)
                        # EW light red
                        pygame.draw.circle(self.screen, (255, 0, 0), 
                                          (light_x + light_radius, light_y), light_radius // 2)
                    else:  # EW Green
                        # NS light red
                        pygame.draw.circle(self.screen, (255, 0, 0), 
                                          (light_x, light_y - light_radius), light_radius // 2)
                        # EW light green
                        pygame.draw.circle(self.screen, (0, 255, 0), 
                                          (light_x + light_radius, light_y), light_radius // 2)
                    
                    # Display traffic density as text
                    try:
                        # Use pre-initialized font
                        if hasattr(self, 'font') and self.font:
                            ns_text = self.font.render(f"NS: {self.traffic_density[idx, 0]:.2f}", True, (0, 0, 0))
                            ew_text = self.font.render(f"EW: {self.traffic_density[idx, 1]:.2f}", True, (0, 0, 0))
                        else:
                            # Fallback to creating a new font
                            font = pygame.font.Font(None, 24)
                            ns_text = font.render(f"NS: {self.traffic_density[idx, 0]:.2f}", True, (0, 0, 0))
                            ew_text = font.render(f"EW: {self.traffic_density[idx, 1]:.2f}", True, (0, 0, 0))
                        
                        self.screen.blit(ns_text, (x + 10, y + 10))
                        self.screen.blit(ew_text, (x + 10, y + 30))
                    except Exception as e:
                        # Continue without text if font rendering fails
                        logger.warning(f"Font rendering failed: {e}")
            
            # Add episode and step information
            if hasattr(self, 'current_episode') and hasattr(self, 'current_step'):
                try:
                    if hasattr(self, 'font') and self.font:
                        info_text = self.font.render(
                            f"Episode: {self.current_episode}, Step: {self.current_step}", 
                            True, (0, 0, 0)
                        )
                        self.screen.blit(info_text, (10, self.screen_height - 30))
                except Exception as e:
                    logger.warning(f"Could not render episode info: {e}")
            
            # Update display
            pygame.display.flip()
            
            # Control frame rate
            self.clock.tick(self.metadata['render_fps'])
            
            if mode == 'rgb_array':
                try:
                    return np.transpose(
                        np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
                    )
                except Exception as e:
                    logger.warning(f"Could not create RGB array: {e}")
                    return None
                    
        except Exception as e:
            logger.error(f"Render failed: {e}")
            self.visualization = False  # Disable visualization after error
            return None
    
    def close(self):
        """Close the environment."""
        if self.visualization:
            pygame.quit()