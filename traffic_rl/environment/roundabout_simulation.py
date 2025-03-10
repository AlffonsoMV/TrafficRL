"""
Roundabout Traffic Simulation Environment
======================================
A custom Gym environment for roundabout traffic simulation.
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import logging
import math
from traffic_rl.environment.traffic_simulation import TrafficSimulation

logger = logging.getLogger("TrafficRL.Environment.Roundabout")

class RoundaboutSimulation(TrafficSimulation):
    """
    Custom Gym environment for roundabout traffic simulation.
    
    Represents a roundabout with multiple entry/exit points controlled by traffic lights.
    Each entry point has two incoming lanes (Entry and Exit).
    """
    
    def __init__(self, config, visualization=False, random_seed=None):
        """
        Initialize the roundabout simulation environment.
        
        Args:
            config: Configuration dictionary
            visualization: Whether to enable visualization
            random_seed: Random seed for reproducibility
        """
        # Call parent constructor but we'll override some methods
        super(RoundaboutSimulation, self).__init__(config, visualization, random_seed)
        
        # Override grid_size with number of roundabout entry points
        self.num_entry_points = config.get("num_entry_points", 4)
        
        # Number of intersections is now the number of entry points
        self.num_intersections = self.num_entry_points
        
        # Traffic light states: 0=Entry Green (Roundabout Red), 1=Roundabout Green (Entry Red)
        self.action_space = spaces.Discrete(2)
        
        # Observation space: traffic density and light state for each entry point
        # For each entry point: [Entry_density, Roundabout_density, light_state, Entry_waiting, Roundabout_waiting]
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(self.num_intersections, 5),
            dtype=np.float32
        )
        
        # Roundabout specific parameters
        self.roundabout_capacity = config.get("roundabout_capacity", 0.8)  # Max density in the roundabout
        self.roundabout_density = 0.0  # Current density in the roundabout
        
        # Track green light durations for each direction (for roundabout)
        self.entry_green_duration = np.zeros(self.num_intersections)
        self.roundabout_green_duration = np.zeros(self.num_intersections)
        
        # Add these attributes to maintain compatibility with parent class
        self.ns_green_duration = self.entry_green_duration
        self.ew_green_duration = self.roundabout_green_duration
        
        # Reset the environment
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        # Set seed if provided
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        
        # Initialize traffic densities (random initial state)
        # For each entry point: [Entry_density, Exit_density]
        self.traffic_density = self.np_random.uniform(
            0.0, 0.5, size=(self.num_intersections, 2)
        )
        
        # Initialize roundabout density
        self.roundabout_density = self.np_random.uniform(0.0, 0.3)
        
        # Initialize traffic light states (all start with Entry green)
        self.light_states = np.zeros(self.num_intersections, dtype=int)
        
        # Initialize timers for each traffic light
        self.timers = np.zeros(self.num_intersections)
        
        # Track waiting time for cars at each entry point
        self.waiting_time = np.zeros((self.num_intersections, 2))
        
        # Track number of cars passed through each entry point
        self.cars_passed = np.zeros((self.num_intersections, 2))
        
        # Track green light durations for each direction
        self.entry_green_duration = np.zeros(self.num_intersections)
        self.roundabout_green_duration = np.zeros(self.num_intersections)
        
        # Initialize simulation time
        self.current_time = 0.0
        
        # Update these attributes to maintain compatibility with parent class
        self.ns_green_duration = self.entry_green_duration
        self.ew_green_duration = self.roundabout_green_duration
        
        self.light_switches = 0
        
        # Get initial observation
        observation = self._get_observation()
        
        # Info dictionary
        info = {}
        
        return observation, info
    
    def step(self, actions):
        """
        Take a step in the environment given the actions.
        
        Args:
            actions: Array of actions for each entry point (0=Entry Green, 1=Roundabout Green)
                    or a single action to apply to all entry points
        
        Returns:
            observation: Current observation
            reward: Reward from the action
            terminated: Whether the episode is done
            truncated: Whether the episode is truncated
            info: Additional information
        """
        try:
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
            
            # Update traffic lights based on actions
            for i in range(self.num_intersections):
                # Only change the light if the timer has expired
                if self.timers[i] <= 0:
                    # Check if we're switching the light
                    if self.light_states[i] != actions_array[i]:
                        self.light_switches += 1
                    
                    self.light_states[i] = actions_array[i]
                    self.timers[i] = self.green_duration
                else:
                    # Decrease the timer
                    self.timers[i] -= 1
                    
            # Update duration trackers
            for i in range(self.num_intersections):
                if self.light_states[i] == 0:  # Entry is green
                    self.entry_green_duration[i] += 1
                    self.roundabout_green_duration[i] = 0
                else:  # Roundabout is green
                    self.roundabout_green_duration[i] += 1
                    self.entry_green_duration[i] = 0
                
            # Update these attributes to maintain compatibility with parent class
            self.ns_green_duration = self.entry_green_duration
            self.ew_green_duration = self.roundabout_green_duration
                
            # Simulate traffic flow
            self._update_traffic()
            
            # Calculate reward
            reward = self._calculate_reward()
            
            # Generate observation
            observation = self._get_observation()
            
            # Update simulation time
            self.current_time += 1.0
            
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
        """Simulate traffic flow and update densities with roundabout behavior."""
        try:
            # Track pre-update densities
            prev_density = np.copy(self.traffic_density)
            prev_roundabout_density = self.roundabout_density
            
            # Define speed factors based on density (congestion effects)
            # Higher density = slower traffic flow
            speed_factor_entry = 1.0 - 0.7 * prev_density[:, 0]  # Speed factor for entry
            speed_factor_roundabout = 1.0 - 0.7 * prev_roundabout_density  # Speed factor for roundabout
            
            # Base flow rates
            base_flow_rate = 0.1  # Base flow rate with green light
            red_light_flow = 0.01  # Small flow even with red light (running red)
            
            # Track total flow into and out of roundabout
            total_flow_into_roundabout = 0
            total_flow_out_of_roundabout = 0
            
            for i in range(self.num_intersections):
                # Get current light state (0=Entry Green, 1=Roundabout Green)
                light = self.light_states[i]
                
                # Calculate flow rates with congestion effects
                if light == 0:  # Entry green
                    # Green light flow rate affected by congestion and roundabout capacity
                    # Only allow flow into roundabout if it's not at capacity
                    if self.roundabout_density < self.roundabout_capacity:
                        entry_flow_rate = base_flow_rate * speed_factor_entry[i] * (1 - self.roundabout_density / self.roundabout_capacity)
                    else:
                        entry_flow_rate = 0  # Roundabout is full
                        
                    # Small flow from roundabout to exit (some cars exit regardless of light)
                    exit_flow_rate = red_light_flow * speed_factor_roundabout
                else:  # Roundabout green
                    # Small flow from entry to roundabout (some cars enter regardless of light)
                    if self.roundabout_density < self.roundabout_capacity:
                        entry_flow_rate = red_light_flow * speed_factor_entry[i] * (1 - self.roundabout_density / self.roundabout_capacity)
                    else:
                        entry_flow_rate = 0  # Roundabout is full
                        
                    # Green light flow rate from roundabout to exit
                    exit_flow_rate = base_flow_rate * speed_factor_roundabout
                
                # Calculate actual flow based on current density
                entry_cars_flow = min(self.traffic_density[i, 0], entry_flow_rate)
                exit_cars_flow = min(self.roundabout_density / self.num_intersections, exit_flow_rate)
                
                # Update densities and stats
                self.traffic_density[i, 0] -= entry_cars_flow  # Cars leaving entry
                total_flow_into_roundabout += entry_cars_flow  # Cars entering roundabout
                
                # Cars exiting roundabout to exit lane
                self.traffic_density[i, 1] += exit_cars_flow  # Cars entering exit
                total_flow_out_of_roundabout += exit_cars_flow  # Cars leaving roundabout
                
                # Track cars that passed through
                self.cars_passed[i, 0] += entry_cars_flow * self.max_cars
                self.cars_passed[i, 1] += exit_cars_flow * self.max_cars
                
                # Calculate waiting time based on density and whether light is red
                if light == 0:  # Entry Green
                    # Cars in roundabout wait to exit
                    self.waiting_time[i, 1] += self.roundabout_density / self.num_intersections
                else:  # Roundabout Green
                    # Cars at entry wait to enter
                    self.waiting_time[i, 0] += self.traffic_density[i, 0]
            
            # Update roundabout density
            self.roundabout_density += total_flow_into_roundabout - total_flow_out_of_roundabout
            self.roundabout_density = np.clip(self.roundabout_density, 0.0, 1.0)
            
            # Simulate new cars arriving with daily patterns
            # Time of day effect (0=midnight, 0.5=noon, 1.0=midnight again)
            time_of_day = (self.current_time % 1440) / 1440.0  # Normalize to [0,1]
            
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
                # New cars arrive at each entry point with pattern effects
                arrival_factor = base_arrival * (1 + rush_hour_factor)
                entry_arrivals = arrival_factor * self.np_random.uniform(0.5, 1.5)
                
                # Add new cars to entry (ensure density doesn't exceed 1.0)
                self.traffic_density[i, 0] = min(1.0, self.traffic_density[i, 0] + entry_arrivals)
                
                # Process cars leaving exit lanes (they leave the simulation)
                exit_departure_rate = 0.15  # Rate at which cars leave exit lanes
                exit_departures = min(self.traffic_density[i, 1], exit_departure_rate)
                self.traffic_density[i, 1] -= exit_departures
            
            # Simulate traffic flow between adjacent exit and entry points
            # Cars leaving exit can enter the next entry point
            for i in range(self.num_intersections):
                next_i = (i + 1) % self.num_intersections
                
                # Some cars from exit lane go to the next entry point
                transfer_rate = 0.05
                transfer_amount = min(self.traffic_density[i, 1], transfer_rate)
                
                # Transfer cars from exit to next entry
                self.traffic_density[i, 1] -= transfer_amount
                self.traffic_density[next_i, 0] += transfer_amount
                
        except Exception as e:
            logger.error(f"Error in traffic update: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _get_observation(self):
        """
        Construct the observation from the current state.
        
        For each entry point, the observation includes:
        - Entry traffic density (normalized)
        - Roundabout traffic density (normalized)
        - Traffic light state (0 for Entry green, 1 for Roundabout green)
        - Entry waiting time (normalized)
        - Roundabout waiting time (normalized)
        """
        observation = np.zeros((self.num_intersections, 5), dtype=np.float32)
        
        for i in range(self.num_intersections):
            # Traffic density for Entry and Roundabout
            observation[i, 0] = self.traffic_density[i, 0]  # Entry density
            observation[i, 1] = self.roundabout_density  # Roundabout density
            
            # Traffic light state
            observation[i, 2] = self.light_states[i]
            
            # Add waiting time information 
            observation[i, 3] = self.waiting_time[i, 0] / 10.0  # Normalized Entry waiting
            observation[i, 4] = self.waiting_time[i, 1] / 10.0  # Normalized Roundabout waiting
        
        return observation
    
    def render(self, mode='human'):
        """Render the roundabout environment with enhanced visualization."""
        if not self.visualization:
            return None
        
        try:
            # Check if pygame is still running
            if not pygame.get_init():
                logger.warning("Pygame not initialized, reinitializing...")
                self._init_visualization()
                
            # Fill background with a light gray color for better contrast
            self.screen.fill((240, 240, 240))
            
            # Calculate center and radius of roundabout
            center_x = self.screen_width // 2
            center_y = self.screen_height // 2
            # Reduce the outer radius to leave more space for the dashboard
            outer_radius = min(self.screen_width, self.screen_height) // 3.5
            inner_radius = outer_radius // 2
            
            # Draw roundabout with improved visuals
            # Outer circle (asphalt)
            pygame.draw.circle(self.screen, (80, 80, 80), (center_x, center_y), outer_radius)
            # Inner circle (grass/center island)
            pygame.draw.circle(self.screen, (100, 180, 100), (center_x, center_y), inner_radius)
            # Lane markings (dashed white line)
            mid_radius = (outer_radius + inner_radius) // 2
            for i in range(36):
                if i % 2 == 0:  # Draw only every other segment for dashed line
                    start_angle = 2 * math.pi * i / 36
                    end_angle = 2 * math.pi * (i + 0.5) / 36
                    start_x = center_x + int(mid_radius * math.cos(start_angle))
                    start_y = center_y + int(mid_radius * math.sin(start_angle))
                    end_x = center_x + int(mid_radius * math.cos(end_angle))
                    end_y = center_y + int(mid_radius * math.sin(end_angle))
                    pygame.draw.line(self.screen, (255, 255, 255), (start_x, start_y), (end_x, end_y), 2)
            
            # Draw entry/exit points and traffic lights
            for i in range(self.num_intersections):
                # Calculate position (evenly spaced around the circle)
                angle = 2 * math.pi * i / self.num_intersections
                
                # Entry/exit point position (on the outer circle)
                entry_x = center_x + int(outer_radius * math.cos(angle))
                entry_y = center_y + int(outer_radius * math.sin(angle))
                
                # Draw road from outside to roundabout - WIDER ROADS
                # Increase road width by 50%
                road_width = int(outer_radius // 6)  # Was outer_radius // 8
                
                # Calculate road endpoints - extend roads further out
                outer_x = center_x + int((outer_radius + outer_radius * 0.7) * math.cos(angle))
                outer_y = center_y + int((outer_radius + outer_radius * 0.7) * math.sin(angle))
                
                # Draw entry/exit roads with lane markings
                pygame.draw.line(self.screen, (80, 80, 80), 
                                (outer_x, outer_y), (entry_x, entry_y), road_width)
                
                # Add lane markings on roads
                lane_x1 = entry_x + int((outer_x - entry_x) * 0.25)
                lane_y1 = entry_y + int((outer_y - entry_y) * 0.25)
                lane_x2 = entry_x + int((outer_x - entry_x) * 0.75)
                lane_y2 = entry_y + int((outer_y - entry_y) * 0.75)
                pygame.draw.line(self.screen, (255, 255, 255), 
                                (lane_x1, lane_y1), (lane_x2, lane_y2), 2)
                
                # Draw traffic light with improved visuals
                light_radius = road_width // 2
                light_x = entry_x - int(road_width * 0.8 * math.cos(angle))
                light_y = entry_y - int(road_width * 0.8 * math.sin(angle))
                
                # Traffic light housing
                pygame.draw.circle(self.screen, (50, 50, 50), (light_x, light_y), light_radius)
                
                # Traffic light color
                if self.light_states[i] == 0:  # Entry Green
                    # Entry light green
                    pygame.draw.circle(self.screen, (0, 255, 0), 
                                      (light_x, light_y), light_radius // 1.5)
                else:  # Roundabout Green
                    # Entry light red
                    pygame.draw.circle(self.screen, (255, 0, 0), 
                                      (light_x, light_y), light_radius // 1.5)
                
                # Visualize traffic density with cars
                num_cars_entry = int(self.traffic_density[i, 0] * 10)
                num_cars_exit = int(self.traffic_density[i, 1] * 10)
                
                # Draw cars on entry roads
                for j in range(num_cars_entry):
                    car_distance = j * road_width / 10
                    car_x = entry_x + int((outer_x - entry_x) * (0.2 + car_distance / (road_width * 1.5)))
                    car_y = entry_y + int((outer_y - entry_y) * (0.2 + car_distance / (road_width * 1.5)))
                    pygame.draw.circle(self.screen, (255, 0, 0), (car_x, car_y), 5)  # Red cars for entry, slightly larger
                
                # Draw cars on exit roads
                for j in range(num_cars_exit):
                    car_distance = j * road_width / 10
                    car_x = entry_x + int((outer_x - entry_x) * (0.8 - car_distance / (road_width * 1.5)))
                    car_y = entry_y + int((outer_y - entry_y) * (0.8 - car_distance / (road_width * 1.5)))
                    pygame.draw.circle(self.screen, (0, 0, 255), (car_x, car_y), 5)  # Blue cars for exit, slightly larger
                
                # Display traffic density as text with improved formatting
                try:
                    # Use pre-initialized font
                    if hasattr(self, 'font') and self.font:
                        entry_text = self.font.render(f"Entry {i+1}: {self.traffic_density[i, 0]:.2f}", True, (0, 0, 0))
                        exit_text = self.font.render(f"Exit {i+1}: {self.traffic_density[i, 1]:.2f}", True, (0, 0, 0))
                        wait_text = self.font.render(f"Wait: {self.waiting_time[i, 0]:.1f}s", True, (0, 0, 0))
                    else:
                        # Fallback to creating a new font
                        font = pygame.font.Font(None, 24)
                        entry_text = font.render(f"Entry {i+1}: {self.traffic_density[i, 0]:.2f}", True, (0, 0, 0))
                        exit_text = font.render(f"Exit {i+1}: {self.traffic_density[i, 1]:.2f}", True, (0, 0, 0))
                        wait_text = font.render(f"Wait: {self.waiting_time[i, 0]:.1f}s", True, (0, 0, 0))
                    
                    # Calculate text position - move text further out to accommodate wider roads
                    text_angle = angle
                    text_distance = outer_radius + 80  # Increased from 50
                    text_x = center_x + int(text_distance * math.cos(text_angle))
                    text_y = center_y + int(text_distance * math.sin(text_angle))
                    
                    # Create a semi-transparent background for text
                    text_bg_width = 120
                    text_bg_height = 70
                    text_bg = pygame.Surface((text_bg_width, text_bg_height), pygame.SRCALPHA)
                    text_bg.fill((255, 255, 255, 180))  # Semi-transparent white
                    self.screen.blit(text_bg, (text_x - 60, text_y - 25))
                    
                    self.screen.blit(entry_text, (text_x - 55, text_y - 20))
                    self.screen.blit(exit_text, (text_x - 55, text_y))
                    self.screen.blit(wait_text, (text_x - 55, text_y + 20))
                except Exception as e:
                    # Continue without text if font rendering fails
                    logger.warning(f"Font rendering failed: {e}")
            
            # Visualize cars in the roundabout
            num_roundabout_cars = int(self.roundabout_density * 20)
            for i in range(num_roundabout_cars):
                car_angle = 2 * math.pi * i / num_roundabout_cars
                car_radius = (inner_radius + outer_radius) / 2
                car_x = center_x + int(car_radius * math.cos(car_angle))
                car_y = center_y + int(car_radius * math.sin(car_angle))
                pygame.draw.circle(self.screen, (0, 200, 200), (car_x, car_y), 5)  # Cyan cars for roundabout
            
            # Display roundabout density with improved formatting
            try:
                if hasattr(self, 'font') and self.font:
                    # Create a semi-transparent background for roundabout info
                    roundabout_bg = pygame.Surface((160, 30), pygame.SRCALPHA)
                    roundabout_bg.fill((255, 255, 255, 180))  # Semi-transparent white
                    self.screen.blit(roundabout_bg, (center_x - 80, center_y - 15))
                    
                    roundabout_text = self.font.render(f"Roundabout: {self.roundabout_density:.2f}", True, (0, 0, 0))
                    self.screen.blit(roundabout_text, (center_x - 75, center_y - 10))
            except Exception as e:
                logger.warning(f"Font rendering failed: {e}")
            
            # Create a dashboard at the top of the screen with improved aesthetics
            dashboard_height = 50  # Further reduced from 60 to 50
            dashboard_width = self.screen_width
            
            # Create a gradient background for the dashboard
            dashboard = pygame.Surface((dashboard_width, dashboard_height), pygame.SRCALPHA)
            
            # Create a gradient from dark blue to dark purple
            for y in range(dashboard_height):
                # Calculate gradient color (dark blue to dark purple)
                r = int(40 + (y / dashboard_height) * 20)  # 40-60
                g = int(45 + (y / dashboard_height) * 10)  # 45-55
                b = int(80 + (y / dashboard_height) * 40)  # 80-120
                pygame.draw.line(dashboard, (r, g, b, 230), (0, y), (dashboard_width, y))
            
            # Add a subtle bottom border
            pygame.draw.line(dashboard, (120, 120, 160, 255), 
                            (0, dashboard_height-1), (dashboard_width, dashboard_height-1), 2)
            
            self.screen.blit(dashboard, (0, 0))  # Position at the top (0, 0)
            
            # Add epoch information and metrics to dashboard with improved styling
            try:
                if hasattr(self, 'font') and self.font:
                    # Create fonts for the dashboard
                    dashboard_font = pygame.font.Font(None, 24)  # Main font
                    
                    # Define sections with equal width
                    section_width = dashboard_width // 3
                    
                    # Episode and step information (left section)
                    if hasattr(self, 'current_episode') and hasattr(self, 'current_step'):
                        # Create a section background
                        left_section = pygame.Surface((section_width, dashboard_height), pygame.SRCALPHA)
                        left_section.fill((255, 255, 255, 15))  # Very subtle white overlay
                        self.screen.blit(left_section, (0, 0))
                        
                        # Add a subtle vertical separator
                        pygame.draw.line(self.screen, (150, 150, 180, 100), 
                                        (section_width-1, 5), (section_width-1, dashboard_height-5), 1)
                        
                        # Episode text with shadow effect
                        episode_text = dashboard_font.render(
                            f"Episode: {self.current_episode}", 
                            True, (255, 255, 255)
                        )
                        # Add shadow
                        episode_shadow = dashboard_font.render(
                            f"Episode: {self.current_episode}", 
                            True, (20, 20, 40)
                        )
                        self.screen.blit(episode_shadow, (22, 12))  # Shadow
                        self.screen.blit(episode_text, (20, 10))  # Text
                        
                        # Step text
                        step_text = dashboard_font.render(
                            f"Step: {self.current_step}", 
                            True, (220, 220, 255)
                        )
                        self.screen.blit(step_text, (20, 30))
                    
                    # Traffic pattern information (middle section)
                    if hasattr(self, 'traffic_pattern'):
                        # Create a section background
                        middle_section = pygame.Surface((section_width, dashboard_height), pygame.SRCALPHA)
                        middle_section.fill((255, 255, 255, 15))  # Very subtle white overlay
                        self.screen.blit(middle_section, (section_width, 0))
                        
                        # Add a subtle vertical separator
                        pygame.draw.line(self.screen, (150, 150, 180, 100), 
                                        (section_width*2-1, 5), (section_width*2-1, dashboard_height-5), 1)
                        
                        # Pattern text with icon
                        pattern_text = dashboard_font.render(
                            f"Pattern: {self.traffic_pattern.capitalize()}", 
                            True, (255, 255, 255)
                        )
                        # Add shadow
                        pattern_shadow = dashboard_font.render(
                            f"Pattern: {self.traffic_pattern.capitalize()}", 
                            True, (20, 20, 40)
                        )
                        self.screen.blit(pattern_shadow, (section_width + 22, 12))  # Shadow
                        self.screen.blit(pattern_text, (section_width + 20, 10))
                    
                    # Average waiting time with colored indicator
                    avg_waiting = self.waiting_time.mean() if hasattr(self, 'waiting_time') else 0
                    
                    # Determine color based on waiting time (green to red)
                    if avg_waiting < 5:
                        wait_color = (100, 255, 100)  # Green for low waiting time
                    elif avg_waiting < 15:
                        wait_color = (255, 255, 100)  # Yellow for medium waiting time
                    else:
                        wait_color = (255, 100, 100)  # Red for high waiting time
                        
                    waiting_text = dashboard_font.render(
                        f"Avg Wait: {avg_waiting:.1f}s", 
                        True, wait_color
                    )
                    self.screen.blit(waiting_text, (section_width + 20, 30))
                    
                    # Right section - metrics
                    if hasattr(self, 'cars_passed') and hasattr(self, 'current_time'):
                        # Create a section background
                        right_section = pygame.Surface((section_width, dashboard_height), pygame.SRCALPHA)
                        right_section.fill((255, 255, 255, 15))  # Very subtle white overlay
                        self.screen.blit(right_section, (section_width*2, 0))
                        
                        # Total cars passed
                        total_cars = self.cars_passed.sum()
                        cars_text = dashboard_font.render(
                            f"Cars Passed: {int(total_cars)}", 
                            True, (255, 255, 255)
                        )
                        # Add shadow
                        cars_shadow = dashboard_font.render(
                            f"Cars Passed: {int(total_cars)}", 
                            True, (20, 20, 40)
                        )
                        self.screen.blit(cars_shadow, (section_width*2 + 22, 12))  # Shadow
                        self.screen.blit(cars_text, (section_width*2 + 20, 10))
                        
                        # Current time with clock icon
                        time_text = dashboard_font.render(
                            f"Time: {int(self.current_time)}s", 
                            True, (220, 220, 255)
                        )
                        self.screen.blit(time_text, (section_width*2 + 20, 30))
                        
            except Exception as e:
                logger.warning(f"Could not render dashboard: {e}")
            
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