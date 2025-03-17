"""
Car Entity Module
================
Defines car entities for more realistic traffic simulation.
"""

import numpy as np
import logging
from enum import Enum, auto

logger = logging.getLogger("TrafficRL.Car")

class Direction(Enum):
    """Enum for car directions"""
    NORTH = auto()
    EAST = auto()
    SOUTH = auto()
    WEST = auto()
    
    @staticmethod
    def is_north_south(direction):
        """Check if direction is north or south"""
        return direction in (Direction.NORTH, Direction.SOUTH)
    
    @staticmethod
    def is_east_west(direction):
        """Check if direction is east or west"""
        return direction in (Direction.EAST, Direction.WEST)
    
    @staticmethod
    def get_opposite(direction):
        """Get the opposite direction"""
        if direction == Direction.NORTH:
            return Direction.SOUTH
        elif direction == Direction.SOUTH:
            return Direction.NORTH
        elif direction == Direction.EAST:
            return Direction.WEST
        elif direction == Direction.WEST:
            return Direction.EAST


class CarState(Enum):
    """Enum for car states"""
    MOVING = auto()      # Moving normally
    WAITING = auto()     # Waiting at red light or traffic
    TURNING = auto()     # In the process of turning
    EXITING = auto()     # Exiting the grid


class Car:
    """
    Represents an individual car in the traffic simulation.
    
    Each car has a position, direction, speed, and destination.
    Cars can make decisions about acceleration, lane changes, and turns
    based on traffic conditions and their destination.
    """
    
    # Class variable to track total cars created
    _next_id = 0
    
    def __init__(self, position, direction, speed=1.0, grid_size=4, np_random=None):
        """
        Initialize a car entity.
        
        Args:
            position: Tuple of (x, y) coordinates
            direction: Direction the car is traveling
            speed: Initial speed (default 1.0)
            grid_size: Size of the traffic grid
            np_random: NumPy random generator
        """
        # Assign unique ID to this car
        self.id = Car._next_id
        Car._next_id += 1
        
        # Position, speed and direction
        self.position = position
        self.direction = direction
        self.speed = speed
        self.max_speed = 1.5
        
        # Track waiting time
        self.waiting_time = 0
        self.total_waiting_time = 0
        self.state = CarState.MOVING
        self.time_since_last_move = 0
        
        # Store grid size for boundary checks
        self.grid_size = grid_size
        
        # Random number generator
        self.np_random = np_random if np_random is not None else np.random
        
        # Determine destination - cars generally have a target exit point
        self._set_destination()
        
        # Set turning probabilities based on destination
        self._set_turning_probabilities()
        
        # Decision-making attributes
        self.patience = self.np_random.uniform(0.5, 1.0)  # Varies by driver
        self.aggressiveness = self.np_random.uniform(0.1, 0.9)  # Varies by driver
        self.next_turn_direction = None
        
        # For visualization
        self.color = tuple(self.np_random.randint(50, 250, 3))  # Random color
    
    def _set_destination(self):
        """Set a destination for the car (typically an exit point on the grid)"""
        # Most cars want to go somewhere specific, not just wander
        # Determine an exit point on the edge of the grid
        edge_points = []
        
        # Add possible edge coordinates based on current direction
        # Cars are more likely to continue in their general direction
        if self.direction == Direction.NORTH:
            # More likely to exit from the north or east/west, less likely south
            edge_points += [(x, 0) for x in range(self.grid_size)]  # North edge
            edge_points += [(0, y) for y in range(self.grid_size)]  # West edge
            edge_points += [(self.grid_size-1, y) for y in range(self.grid_size)]  # East edge
        elif self.direction == Direction.SOUTH:
            # More likely to exit from the south or east/west, less likely north
            edge_points += [(x, self.grid_size-1) for x in range(self.grid_size)]  # South edge
            edge_points += [(0, y) for y in range(self.grid_size)]  # West edge
            edge_points += [(self.grid_size-1, y) for y in range(self.grid_size)]  # East edge
        elif self.direction == Direction.EAST:
            # More likely to exit from the east or north/south, less likely west
            edge_points += [(self.grid_size-1, y) for y in range(self.grid_size)]  # East edge
            edge_points += [(x, 0) for x in range(self.grid_size)]  # North edge
            edge_points += [(x, self.grid_size-1) for x in range(self.grid_size)]  # South edge
        elif self.direction == Direction.WEST:
            # More likely to exit from the west or north/south, less likely east
            edge_points += [(0, y) for y in range(self.grid_size)]  # West edge
            edge_points += [(x, 0) for x in range(self.grid_size)]  # North edge
            edge_points += [(x, self.grid_size-1) for x in range(self.grid_size)]  # South edge
            
        # Filter out current position if it's on an edge
        edge_points = [edge for edge in edge_points if edge != self.position]
        
        if edge_points:
            # Convert to numpy array for proper indexing
            edge_points = np.array(edge_points)
            # Choose a random index and retrieve the destination
            idx = self.np_random.randint(0, len(edge_points))
            self.destination = tuple(edge_points[idx])
        else:
            # Fallback - random edge
            x = self.np_random.randint(0, self.grid_size)
            y = self.np_random.randint(0, self.grid_size)
            
            # Ensure it's on the edge
            if self.np_random.random() < 0.5:
                x = 0 if self.np_random.random() < 0.5 else self.grid_size - 1
            else:
                y = 0 if self.np_random.random() < 0.5 else self.grid_size - 1
                
            self.destination = (x, y)
    
    def _set_turning_probabilities(self):
        """Set the probability of turning at each intersection based on destination"""
        # Determine direction to destination
        dest_x, dest_y = self.destination
        curr_x, curr_y = self.position
        
        # Direction to destination (simplified)
        dx = dest_x - curr_x
        dy = dest_y - curr_y
        
        # Initialize turning probabilities
        # Default values - slight preference for going straight
        self.turn_prob = {
            'left': 0.1,
            'right': 0.1,
            'straight': 0.8
        }
        
        # Adjust based on destination
        if abs(dx) > abs(dy):
            # Destination is more east/west
            if dx > 0:  # East
                if self.direction == Direction.NORTH:
                    self.turn_prob = {'left': 0.1, 'right': 0.7, 'straight': 0.2}
                elif self.direction == Direction.SOUTH:
                    self.turn_prob = {'left': 0.7, 'right': 0.1, 'straight': 0.2}
                elif self.direction == Direction.EAST:
                    self.turn_prob = {'left': 0.1, 'right': 0.1, 'straight': 0.8}
                elif self.direction == Direction.WEST:
                    self.turn_prob = {'left': 0.4, 'right': 0.4, 'straight': 0.2}
            else:  # West
                if self.direction == Direction.NORTH:
                    self.turn_prob = {'left': 0.7, 'right': 0.1, 'straight': 0.2}
                elif self.direction == Direction.SOUTH:
                    self.turn_prob = {'left': 0.1, 'right': 0.7, 'straight': 0.2}
                elif self.direction == Direction.EAST:
                    self.turn_prob = {'left': 0.4, 'right': 0.4, 'straight': 0.2}
                elif self.direction == Direction.WEST:
                    self.turn_prob = {'left': 0.1, 'right': 0.1, 'straight': 0.8}
        else:
            # Destination is more north/south
            if dy > 0:  # South
                if self.direction == Direction.NORTH:
                    self.turn_prob = {'left': 0.4, 'right': 0.4, 'straight': 0.2}
                elif self.direction == Direction.SOUTH:
                    self.turn_prob = {'left': 0.1, 'right': 0.1, 'straight': 0.8}
                elif self.direction == Direction.EAST:
                    self.turn_prob = {'left': 0.7, 'right': 0.1, 'straight': 0.2}
                elif self.direction == Direction.WEST:
                    self.turn_prob = {'left': 0.1, 'right': 0.7, 'straight': 0.2}
            else:  # North
                if self.direction == Direction.NORTH:
                    self.turn_prob = {'left': 0.1, 'right': 0.1, 'straight': 0.8}
                elif self.direction == Direction.SOUTH:
                    self.turn_prob = {'left': 0.4, 'right': 0.4, 'straight': 0.2}
                elif self.direction == Direction.EAST:
                    self.turn_prob = {'left': 0.1, 'right': 0.7, 'straight': 0.2}
                elif self.direction == Direction.WEST:
                    self.turn_prob = {'left': 0.7, 'right': 0.1, 'straight': 0.2}
    
    def update_position(self, sim_env):
        """
        Update the car's position based on its current state, speed, and environment.
        
        Args:
            sim_env: The traffic simulation environment containing intersections and other cars
            
        Returns:
            bool: True if car is still in the grid, False if it has exited
        """
        if self.state == CarState.EXITING:
            return False
            
        # Get current grid cell/intersection
        x, y = self.position
        grid_x, grid_y = int(x), int(y)
        
        # Check if we're at an intersection (when position is close to integer coordinates)
        at_intersection = (abs(x - grid_x) < 0.1 and abs(y - grid_y) < 0.1)
        
        # Track time since last movement
        if self.speed < 0.1:  # If essentially stopped
            self.time_since_last_move += 1
            if self.time_since_last_move > 3:  # If stuck for too long
                # Slightly increase chance of moving to prevent deadlock
                self.speed = max(0.1, self.speed)
        else:
            self.time_since_last_move = 0
        
        # Determine next position based on current direction and speed
        next_x, next_y = x, y
        
        # Decision making at intersection
        if at_intersection:
            # Check if we've reached the destination
            if (grid_x, grid_y) == self.destination:
                self.state = CarState.EXITING
                return False
                
            # Check traffic light at this intersection
            intersection_idx = grid_y * self.grid_size + grid_x
            light_state = sim_env.light_states[intersection_idx]  # 0=NS Green, 1=EW Green
            
            # Check if we need to stop for red light
            must_stop = False
            
            # N/S directions stop on EW green (light_state == 1)
            # E/W directions stop on NS green (light_state == 0)
            if Direction.is_north_south(self.direction) and light_state == 1:
                must_stop = True
            elif Direction.is_east_west(self.direction) and light_state == 0:
                must_stop = True
            
            # If at red light, wait
            if must_stop:
                self.speed = 0
                self.state = CarState.WAITING
                self.waiting_time += 1
                self.total_waiting_time += 1
                return True
                
            # At green light, check if we should turn
            if self.next_turn_direction is None:
                # Decide whether to turn at this intersection
                turn_decision = self.np_random.choice(
                    ['left', 'right', 'straight'], 
                    p=[self.turn_prob['left'], self.turn_prob['right'], self.turn_prob['straight']]
                )
                
                if turn_decision == 'left':
                    # Left turn
                    if self.direction == Direction.NORTH:
                        self.next_turn_direction = Direction.WEST
                    elif self.direction == Direction.EAST:
                        self.next_turn_direction = Direction.NORTH
                    elif self.direction == Direction.SOUTH:
                        self.next_turn_direction = Direction.EAST
                    elif self.direction == Direction.WEST:
                        self.next_turn_direction = Direction.SOUTH
                elif turn_decision == 'right':
                    # Right turn
                    if self.direction == Direction.NORTH:
                        self.next_turn_direction = Direction.EAST
                    elif self.direction == Direction.EAST:
                        self.next_turn_direction = Direction.SOUTH
                    elif self.direction == Direction.SOUTH:
                        self.next_turn_direction = Direction.WEST
                    elif self.direction == Direction.WEST:
                        self.next_turn_direction = Direction.NORTH
                else:
                    # Straight ahead
                    self.next_turn_direction = self.direction
                
                # After crossing the intersection, update direction and reset
                self.direction = self.next_turn_direction
                self.next_turn_direction = None
                
                # Update turning probabilities based on new direction
                self._set_turning_probabilities()
                
            # Set state to moving
            self.state = CarState.MOVING
        
        # Calculate movement based on direction
        if self.direction == Direction.NORTH:
            next_y = y - self.speed
        elif self.direction == Direction.EAST:
            next_x = x + self.speed
        elif self.direction == Direction.SOUTH:
            next_y = y + self.speed
        elif self.direction == Direction.WEST:
            next_x = x - self.speed
        
        # Check if we've left the grid
        if next_x < 0 or next_x >= self.grid_size or next_y < 0 or next_y >= self.grid_size:
            self.state = CarState.EXITING
            return False
        
        # Update position
        self.position = (next_x, next_y)
        
        # If we were waiting but now moving, reset waiting time
        if self.state == CarState.WAITING and self.speed > 0:
            self.state = CarState.MOVING
            self.waiting_time = 0
            
        return True
    
    def adjust_speed(self, sim_env, all_cars):
        """
        Adjust car speed based on traffic conditions.
        
        Args:
            sim_env: The traffic simulation environment
            all_cars: List of all cars in the simulation
        """
        # Default acceleration/deceleration rates
        acceleration = 0.05
        deceleration = 0.1
        
        # Current position and direction
        x, y = self.position
        
        # Check for cars ahead in same direction
        car_ahead = False
        min_distance = float('inf')
        
        # Only consider cars that are fairly close
        nearby_cars = [car for car in all_cars if car.id != self.id]
        
        for other_car in nearby_cars:
            if other_car.direction != self.direction:
                continue
                
            ox, oy = other_car.position
            
            # Distance calculation depends on direction
            if self.direction == Direction.NORTH and oy < y:
                distance = y - oy
                if distance < min_distance:
                    min_distance = distance
                    car_ahead = True
            elif self.direction == Direction.EAST and ox > x:
                distance = ox - x
                if distance < min_distance:
                    min_distance = distance
                    car_ahead = True
            elif self.direction == Direction.SOUTH and oy > y:
                distance = oy - y
                if distance < min_distance:
                    min_distance = distance
                    car_ahead = True
            elif self.direction == Direction.WEST and ox < x:
                distance = x - ox
                if distance < min_distance:
                    min_distance = distance
                    car_ahead = True
        
        # Adjust speed based on car ahead
        if car_ahead and min_distance < 0.5:
            # Slow down if too close to car ahead
            self.speed = max(0, self.speed - deceleration)
        elif self.state != CarState.WAITING:
            # Accelerate if no obstacles ahead
            target_speed = self.max_speed * (0.8 + 0.2 * self.aggressiveness)
            if self.speed < target_speed:
                self.speed = min(target_speed, self.speed + acceleration)
    
    def get_info(self):
        """Return car information for visualization and debugging"""
        return {
            'id': self.id,
            'position': self.position,
            'direction': self.direction,
            'speed': self.speed,
            'state': self.state,
            'waiting_time': self.waiting_time,
            'total_waiting_time': self.total_waiting_time,
            'color': self.color
        }
