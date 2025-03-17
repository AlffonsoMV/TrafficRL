"""
Traffic Simulation Environment
============================
A custom Gym environment for traffic simulation with individual car entities.
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import logging
import traceback
from collections import defaultdict

from traffic_rl.environment.car import Car, Direction, CarState

logger = logging.getLogger("TrafficRL.Environment")

class TrafficSimulation(gym.Env):
    """
    Custom Gym environment for traffic simulation with individual car entities.
    
    Represents a grid of intersections controlled by traffic lights.
    Each intersection has four incoming lanes (North, East, South, West).
    Cars move individually through the grid with their own decision-making logic.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 180}
    
    def __init__(self, config, visualization=False, random_seed=None):
        """
        Initialize the traffic simulation environment.
        
        Args:
            config: Configuration dictionary
            visualization: Whether to enable visualization
            random_seed: Random seed for reproducibility
        """
        super(TrafficSimulation, self).__init__()
        
        # Store configuration
        self.config = config
        
        # Environment configuration
        self.grid_size = config.get("grid_size", 4)
        self.max_cars = config.get("max_cars", 50)
        self.green_duration = config.get("green_duration", 10)
        self.yellow_duration = config.get("yellow_duration", 3)
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
        
        # Observation space: aggregated metrics for each intersection
        # For each intersection: 
        # [NS_cars_count, EW_cars_count, light_state, NS_waiting_cars, EW_waiting_cars]
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(self.num_intersections, 5),
            dtype=np.float32
        )
        
        # Cars management
        self.active_cars = []
        self.exited_cars = []
        self.car_spawn_probability = config.get("car_spawn_probability", 0.2)
        self.max_active_cars = config.get("max_active_cars", 200)
        
        # Initialize default traffic pattern
        self.traffic_pattern = "uniform"
        self.traffic_config = config["traffic_patterns"]["uniform"]
        
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
        
        # Clear cars
        self.active_cars = []
        self.exited_cars = []
        
        # Initialize traffic light states (all start with NS green)
        self.light_states = np.zeros(self.num_intersections, dtype=int)
        
        # Initialize timers for each traffic light
        self.timers = np.zeros(self.num_intersections)
        
        # Track waiting time for cars at each intersection
        self.intersection_waiting_times = np.zeros((self.num_intersections, 2))
        
        # Track number of cars passed through each intersection
        self.cars_passed = np.zeros((self.num_intersections, 2))
        
        # Tracking green light durations
        self.ns_green_duration = np.zeros(self.num_intersections)
        self.ew_green_duration = np.zeros(self.num_intersections)
        self.light_switches = 0
        
        # Statistical trackers
        self.total_cars_generated = 0
        self.total_cars_exited = 0
        self.total_waiting_time = 0
        
        # Simulation time
        self.sim_time = 0
        
        # Generate initial cars (50% of max_active_cars, but at least 100 cars)
        num_initial_cars = min(self.max_active_cars // 2, max(100, self.max_active_cars // 3))
        for _ in range(num_initial_cars):
            self._generate_car()
        
        # Generate observation
        observation = self._get_observation()
        
        # Info dictionary
        info = {}
        
        return observation, info
    
    def step(self, actions):
        """
        Take a step in the environment by applying the traffic light actions.
        
        Args:
            actions: Actions to apply (either single action or list)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        try:
            # Convert single action to array format
            if np.isscalar(actions):
                actions_array = np.full(self.num_intersections, actions)
            else:
                actions_array = np.array(actions)
            
            # Ensure the actions are valid
            actions_array = np.clip(actions_array, 0, self.action_space.n - 1)
            
            # Count light switches
            for i in range(self.num_intersections):
                if actions_array[i] != self.light_states[i] and self.timers[i] <= 0:
                    self.light_switches += 1
            
            # Update traffic lights based on actions
            for i in range(self.num_intersections):
                # Only change the light if the timer has expired
                if self.timers[i] <= 0:
                    self.light_states[i] = actions_array[i]
                    self.timers[i] = self.green_duration
                else:
                    # Decrease the timer
                    self.timers[i] -= 1
                    
            # Update duration trackers
            for i in range(self.num_intersections):
                if self.light_states[i] == 0:  # NS is green
                    self.ns_green_duration[i] += 1
                    self.ew_green_duration[i] = 0
                else:  # EW is green
                    self.ew_green_duration[i] += 1
                    self.ns_green_duration[i] = 0
            
            # Reset intersection waiting time counters for this step
            self.intersection_waiting_times = np.zeros((self.num_intersections, 2))
            
            # Update cars (move them, check for collisions, etc.)
            self._update_cars()
            
            # Generate new cars based on traffic patterns
            self._generate_new_cars()
            
            # Clear any cars that have exited
            self._process_exited_cars()
            
            # Gather statistics for info
            avg_waiting_time = 0
            if self.active_cars:
                avg_waiting_time = sum(car.waiting_time for car in self.active_cars) / len(self.active_cars)
            
            total_cars_count = len(self.active_cars)
            
            # Additional info
            info = {
                'average_waiting_time': avg_waiting_time,
                'total_cars_passed': np.sum(self.cars_passed),
                'active_cars': total_cars_count,
                'total_cars_generated': self.total_cars_generated,
                'total_cars_exited': self.total_cars_exited
            }
            
            # Choose reward function based on config
            use_normalized_reward = self.config.get("use_normalized_reward", False)
            if use_normalized_reward:
                reward = self.compute_simple_reward(info)
            else:
                # Calculate reward using the individual car metrics
                reward = self._calculate_reward()
            
            # Generate observation
            observation = self._get_observation()
            
            # Update simulation time
            self.sim_time += 1
            
            # Check if episode is done (for now, never terminates)
            terminated = False
            truncated = False
            
            # Render if visualization is enabled
            if self.visualization:
                self.render()
            
            return observation, reward, terminated, truncated, info
            
        except Exception as e:
            logger.error(f"Error in step() method: {e}")
            logger.error(traceback.format_exc())
            return self._get_observation(), 0, True, False, {}
    
    def _generate_car(self):
        """Generate a new car at a random entry point on the grid edge."""
        if len(self.active_cars) >= self.max_active_cars:
            return None
            
        # Choose a random edge of the grid
        edge = self.np_random.choice(['north', 'east', 'south', 'west'])
        
        # Determine the position and direction based on the edge
        if edge == 'north':
            x = self.np_random.randint(0, self.grid_size)
            y = 0
            direction = Direction.SOUTH
        elif edge == 'east':
            x = self.grid_size - 1
            y = self.np_random.randint(0, self.grid_size)
            direction = Direction.WEST
        elif edge == 'south':
            x = self.np_random.randint(0, self.grid_size)
            y = self.grid_size - 1
            direction = Direction.NORTH
        else:  # west
            x = 0
            y = self.np_random.randint(0, self.grid_size)
            direction = Direction.EAST
        
        # Create a new car
        car = Car(position=(x, y), direction=direction, 
                 grid_size=self.grid_size, np_random=self.np_random)
        
        # Check for collision with existing cars
        for existing_car in self.active_cars:
            ex, ey = existing_car.position
            if abs(ex - x) < 0.2 and abs(ey - y) < 0.2:
                # Too close to an existing car, don't add this one
                return None
        
        # Add to active cars list
        self.active_cars.append(car)
        self.total_cars_generated += 1
        
        return car
    
    def _generate_new_cars(self):
        """Generate new cars based on traffic patterns and time of day."""
        # Time of day effect (0=midnight, 0.5=noon, 1.0=midnight again)
        time_of_day = (self.sim_time % 1440) / 1440.0  # Normalize to [0,1]
        
        # Get traffic pattern configuration
        if self.traffic_pattern == "natural":
            # Natural traffic pattern with daily and weekly cycles
            # Determine if it's a weekday or weekend
            day_of_week = (self.sim_time // 1440) % 7  # 0=Monday, 6=Sunday
            is_weekend = day_of_week >= 5
            
            # Parameters
            base_prob = self.traffic_config.get("base_arrival", 0.05)
            peak_intensity = self.traffic_config.get("peak_intensity", 2.0)
            weekend_intensity = self.traffic_config.get("weekend_intensity", 1.5)
            
            # Rush hour times
            morning_peak = self.traffic_config.get("morning_peak", 0.33)  # ~8am
            evening_peak = self.traffic_config.get("evening_peak", 0.71)  # ~5pm
            weekend_peak = self.traffic_config.get("weekend_peak", 0.5)   # ~noon
            
            # Night-time reduction factor
            night_factor = 0.3 + 0.7 * (
                np.sin(np.pi * (time_of_day + 0.25)) ** 2  # Higher during day
            )
            
            if is_weekend:
                # Weekend pattern: one main peak around noon
                rush_hour_factor = weekend_intensity * np.exp(-10 * (time_of_day - weekend_peak)**2)
            else:
                # Weekday pattern: morning and evening rush hours
                rush_hour_factor = peak_intensity * (
                    np.exp(-20 * (time_of_day - morning_peak)**2) +  # Morning peak
                    np.exp(-20 * (time_of_day - evening_peak)**2)    # Evening peak
                )
            
            # Apply day/night cycle to the traffic
            spawn_probability = base_prob * (1 + rush_hour_factor * night_factor)
            
        elif self.traffic_pattern == "rush_hour":
            # Morning rush hour ~8am, evening rush hour ~5pm
            morning_peak = self.traffic_config.get("morning_peak", 0.33)
            evening_peak = self.traffic_config.get("evening_peak", 0.71)
            peak_intensity = self.traffic_config.get("peak_intensity", 2.0)
            base_prob = self.traffic_config.get("base_arrival", 0.05)
            
            rush_hour_factor = peak_intensity * (
                np.exp(-20 * (time_of_day - morning_peak)**2) +  # Morning peak
                np.exp(-20 * (time_of_day - evening_peak)**2)    # Evening peak
            )
            
            spawn_probability = base_prob * (1 + rush_hour_factor)
            
        elif self.traffic_pattern == "weekend":
            # Weekend pattern: one peak around noon
            midday_peak = self.traffic_config.get("midday_peak", 0.5)
            peak_intensity = self.traffic_config.get("peak_intensity", 1.5)
            base_prob = self.traffic_config.get("base_arrival", 0.05)
            
            rush_hour_factor = peak_intensity * np.exp(-10 * (time_of_day - midday_peak)**2)
            spawn_probability = base_prob * (1 + rush_hour_factor)
            
        else:  # uniform pattern
            base_prob = self.traffic_config.get("arrival_rate", 0.05)
            spawn_probability = base_prob
        
        # Apply global car spawn probability multiplier
        spawn_probability *= self.car_spawn_probability * 4.0  # Increased from 3.0 to 4.0
        
        # Check if we need to generate more cars to reach target
        current_car_count = len(self.active_cars)
        min_target = 150  # Minimum target of active cars (increased from 100)
        
        # Try to generate cars based on spawn probability
        edges = ['north', 'east', 'south', 'west']
        
        # Generate multiple cars per step if we're below capacity
        # Increase attempts when we have fewer cars than our minimum target
        if current_car_count < min_target:
            num_attempts = min(12, max(8, int((min_target - current_car_count) / 10)))
        else:
            num_attempts = min(6, max(2, int(self.max_active_cars / 50)))
            
        for _ in range(num_attempts):
            for edge in edges:
                if len(self.active_cars) < self.max_active_cars and (
                    len(self.active_cars) < min_target or 
                    self.np_random.random() < spawn_probability
                ):
                    self._generate_car()
    
    def _update_cars(self):
        """Update all active cars positions and states."""
        # First, calculate new speeds for all cars based on their surroundings
        for car in self.active_cars:
            car.adjust_speed(self, self.active_cars)
        
        # Then update positions of all cars
        cars_to_remove = []
        
        for car in self.active_cars:
            # Update car position
            car_in_grid = car.update_position(self)
            
            if not car_in_grid:
                # Car is exiting the grid
                cars_to_remove.append(car)
                
            # Track waiting times at intersections
            if car.state == CarState.WAITING:
                x, y = car.position
                grid_x, grid_y = int(x), int(y)
                intersection_idx = grid_y * self.grid_size + grid_x
                
                # Update waiting time metrics for this intersection
                if Direction.is_north_south(car.direction):
                    self.intersection_waiting_times[intersection_idx, 0] += 1
                else:  # east-west
                    self.intersection_waiting_times[intersection_idx, 1] += 1
        
        # Move exiting cars to the exited list
        for car in cars_to_remove:
            self.active_cars.remove(car)
            self.exited_cars.append(car)
            
            # Track which exit was used for statistics
            x, y = car.position
            grid_x, grid_y = int(min(max(0, x), self.grid_size-1)), int(min(max(0, y), self.grid_size-1))
            intersection_idx = grid_y * self.grid_size + grid_x
            
            # Update cars passed counter
            if Direction.is_north_south(car.direction):
                self.cars_passed[intersection_idx, 0] += 1
            else:  # east-west
                self.cars_passed[intersection_idx, 1] += 1
    
    def _process_exited_cars(self):
        """Process exited cars for statistics."""
        # Count exited cars
        self.total_cars_exited += len(self.exited_cars)
        
        # Calculate total waiting time
        for car in self.exited_cars:
            self.total_waiting_time += car.total_waiting_time
        
        # Clear the exited cars list
        self.exited_cars = []
    
    def _calculate_reward(self):
        """
        Calculate reward based on individual car metrics.
        
        The reward function considers:
        1. Wait time of cars at intersections
        2. Number of cars that successfully exit the grid
        3. Average speed of cars in the system
        4. Fairness of waiting times between different directions
        5. Traffic density and congestion
        """
        try:
            # Get number of active cars
            num_active_cars = len(self.active_cars)
            if num_active_cars == 0:
                return 0.0  # No cars to evaluate
            
            # Calculate average waiting time
            total_waiting = sum(car.waiting_time for car in self.active_cars)
            avg_waiting = total_waiting / num_active_cars if num_active_cars > 0 else 0
            
            # Calculate average speed
            total_speed = sum(car.speed for car in self.active_cars)
            avg_speed = total_speed / num_active_cars if num_active_cars > 0 else 0
            
            # Count cars in each state
            waiting_cars = sum(1 for car in self.active_cars if car.state == CarState.WAITING)
            moving_cars = sum(1 for car in self.active_cars if car.state == CarState.MOVING)
            
            # Calculate ratio of moving cars
            moving_ratio = moving_cars / num_active_cars if num_active_cars > 0 else 0
            
            # Calculate cars exited in this step
            cars_exited = len(self.exited_cars)
            
            # Fairness component: compare NS and EW waiting times
            ns_waiting = sum(1 for car in self.active_cars 
                           if car.state == CarState.WAITING and Direction.is_north_south(car.direction))
            ew_waiting = sum(1 for car in self.active_cars 
                           if car.state == CarState.WAITING and Direction.is_east_west(car.direction))
            
            # Traffic density/congestion factor - sublinear scaling to prevent punishment at high densities
            density_factor = min(1.5, 0.5 + (num_active_cars / self.max_active_cars))
            
            # Improved fairness calculation - weighted by relative traffic volume in each direction
            ns_count = sum(1 for car in self.active_cars if Direction.is_north_south(car.direction))
            ew_count = sum(1 for car in self.active_cars if Direction.is_east_west(car.direction))
            total_count = ns_count + ew_count
            
            if total_count > 0:
                ns_ratio = ns_count / total_count
                ew_ratio = ew_count / total_count
                # Calculate fairness based on relative proportions instead of absolute differences
                # This ensures fairness penalty scales appropriately with traffic
                if ns_count > 0 and ew_count > 0:
                    fairness_penalty = abs((ns_waiting/ns_count) - (ew_waiting/ew_count)) / 2
                else:
                    fairness_penalty = 0
            else:
                fairness_penalty = 0
            
            # Calculate reward components with better balancing
            waiting_penalty = -avg_waiting * 0.05 * density_factor  # Reduced from 0.1 to 0.05
            throughput_reward = cars_exited * 4.0  # Increased from 3.0, don't scale by density
            speed_reward = avg_speed * 5.0
            moving_reward = moving_ratio * 10.0  # Reward for keeping cars moving
            fairness_penalty = -fairness_penalty * 2.0  # Reduced from 3.0 to 2.0
            
            # Add a density management bonus that rewards handling high traffic efficiently
            density_management = (1.0 - (waiting_cars / num_active_cars)) * density_factor * 5.0
            
            # Total reward with rebalanced components
            reward = (
                waiting_penalty +
                throughput_reward +
                speed_reward +
                moving_reward +
                fairness_penalty +
                density_management
            )
            
            # Apply scaling
            if 'reward_scale' in self.config:
                reward *= self.config.get('reward_scale', 1.0)
            
            return reward
            
        except Exception as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0  # Safe default
    
    def _get_observation(self):
        """
        Construct the observation from the current state of individual cars.
        
        For each intersection, aggregates car data into the observation format:
        - NS cars count (normalized)
        - EW cars count (normalized)
        - Traffic light state (0 for NS green, 1 for EW green)
        - NS waiting cars count (normalized)
        - EW waiting cars count (normalized)
        """
        observation = np.zeros((self.num_intersections, 5), dtype=np.float32)
        
        # Count cars near each intersection
        ns_counts = np.zeros(self.num_intersections)
        ew_counts = np.zeros(self.num_intersections)
        ns_waiting = np.zeros(self.num_intersections)
        ew_waiting = np.zeros(self.num_intersections)
        
        # Process each active car
        for car in self.active_cars:
            x, y = car.position
            
            # Find the nearest intersection
            grid_x, grid_y = int(min(max(0, x), self.grid_size-1)), int(min(max(0, y), self.grid_size-1))
            intersection_idx = grid_y * self.grid_size + grid_x
            
            # Update counts based on car direction
            if Direction.is_north_south(car.direction):
                ns_counts[intersection_idx] += 1
                if car.state == CarState.WAITING:
                    ns_waiting[intersection_idx] += 1
            else:  # East-West
                ew_counts[intersection_idx] += 1
                if car.state == CarState.WAITING:
                    ew_waiting[intersection_idx] += 1
        
        # Normalize counts (assuming max 10 cars per direction per intersection)
        max_cars_per_direction = 10
        ns_counts = np.clip(ns_counts / max_cars_per_direction, 0, 1)
        ew_counts = np.clip(ew_counts / max_cars_per_direction, 0, 1)
        ns_waiting = np.clip(ns_waiting / max_cars_per_direction, 0, 1)
        ew_waiting = np.clip(ew_waiting / max_cars_per_direction, 0, 1)
        
        # Construct the observation
        for i in range(self.num_intersections):
            observation[i, 0] = ns_counts[i]
            observation[i, 1] = ew_counts[i]
            observation[i, 2] = self.light_states[i]
            observation[i, 3] = ns_waiting[i]
            observation[i, 4] = ew_waiting[i]
        
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
                
            # Car rendering properties
            self.car_size = 10
                
            logger.info("Visualization initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize visualization: {e}")
            self.visualization = False
            raise
    
    def render(self, mode='human'):
        """Render the environment with individual cars."""
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
            
            # Draw cars
            for car in self.active_cars:
                # Get car position and convert to screen coordinates
                car_x, car_y = car.position
                screen_x = int(car_x * cell_width + cell_width//2)
                screen_y = int(car_y * cell_height + cell_height//2)
                
                # Get car color based on state
                if car.state == CarState.WAITING:
                    color = (255, 0, 0)  # Red for waiting
                elif car.state == CarState.MOVING:
                    color = car.color     # Car's assigned color for moving
                elif car.state == CarState.TURNING:
                    color = (255, 255, 0) # Yellow for turning
                else:
                    color = (100, 100, 100)  # Grey for other states
                
                # Draw the car as a small rectangle
                pygame.draw.rect(self.screen, color, 
                                (screen_x - self.car_size//2, screen_y - self.car_size//2, 
                                 self.car_size, self.car_size))
                
                # Draw a small indicator for car direction
                if car.direction == Direction.NORTH:
                    pygame.draw.line(self.screen, (0, 0, 0),
                                    (screen_x, screen_y), (screen_x, screen_y - self.car_size//2), 2)
                elif car.direction == Direction.EAST:
                    pygame.draw.line(self.screen, (0, 0, 0),
                                    (screen_x, screen_y), (screen_x + self.car_size//2, screen_y), 2)
                elif car.direction == Direction.SOUTH:
                    pygame.draw.line(self.screen, (0, 0, 0),
                                    (screen_x, screen_y), (screen_x, screen_y + self.car_size//2), 2)
                elif car.direction == Direction.WEST:
                    pygame.draw.line(self.screen, (0, 0, 0),
                                    (screen_x, screen_y), (screen_x - self.car_size//2, screen_y), 2)
            
            # Add statistics
            try:
                if hasattr(self, 'font') and self.font:
                    stats_text = self.font.render(
                        f"Active Cars: {len(self.active_cars)}, Exited: {self.total_cars_exited}", 
                        True, (0, 0, 0)
                    )
                    self.screen.blit(stats_text, (10, 10))
                    
                    time_text = self.font.render(
                        f"Sim Time: {self.sim_time}, Day: {self.sim_time // 1440 + 1}, Hour: {(self.sim_time % 1440) // 60}", 
                        True, (0, 0, 0)
                    )
                    self.screen.blit(time_text, (10, 40))
            except Exception as e:
                logger.warning(f"Could not render stats text: {e}")
            
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
            logger.error(traceback.format_exc())
            self.visualization = False  # Disable visualization after error
            return None
    
    def close(self):
        """Close the environment."""
        if self.visualization:
            pygame.quit()

    def compute_simple_reward(self, info):
        """
        Compute a simplified reward based on info dictionary.
        Used primarily for benchmarking different agents with consistent reward calculation.
        
        Args:
            info: Dictionary containing traffic metrics
            
        Returns:
            Float reward value
        """
        # Extract metrics from info
        waiting_time = info.get('avg_waiting_time', 0)
        throughput = info.get('throughput', 0)
        num_cars = info.get('num_cars', 0)
        max_cars = self.config.get('max_active_cars', 300)
        
        # Normalize metrics
        waiting_time_norm = min(1.0, waiting_time / 100.0)  # Cap at 100 seconds
        throughput_norm = min(1.0, throughput / 1000.0)  # Cap at 1000 cars
        
        # Density factor - use sublinear scaling as in _calculate_reward
        density_factor = min(1.5, 0.5 + (num_cars / max_cars))
        
        # Calculate penalties and rewards using the same factors as the main reward
        waiting_penalty = -waiting_time_norm * 5.0 * density_factor  # Scaled down 
        throughput_reward = throughput_norm * 10.0  # Increased weight
        
        # Add bonus for handling high traffic 
        density_management = (1.0 - waiting_time_norm) * density_factor * 3.0
        
        # Calculate total reward
        reward = waiting_penalty + throughput_reward + density_management
        
        # Apply reward scale from config
        reward_scale = self.config.get('reward_scale', 0.01)
        reward = reward * reward_scale
        
        return reward
