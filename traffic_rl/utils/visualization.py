"""
Visualization Module
=================
Visualization utilities for the traffic simulation environment and training results.

This module provides functions for:
1. Visualizing training results (rewards, metrics)
2. Creating video visualizations of the traffic simulation environment
3. Generating static visualizations of traffic patterns and analysis
"""

import os
import numpy as np
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger("TrafficRL")

def visualize_results(rewards_history, avg_rewards_history, car_counts=None, save_path=None):
    """
    Visualize training results.
    
    Args:
        rewards_history: List of episode rewards
        avg_rewards_history: List of average rewards
        car_counts: List of average active car counts per episode
        save_path: Path to save the plot
    """
    try:
        if car_counts:
            # Create a figure with two subplots if we have car counts
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Top plot for rewards
            ax1.plot(rewards_history, alpha=0.6, label='Episode Reward')
            ax1.plot(avg_rewards_history, label='Avg Reward (100 episodes)')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.set_title('Training Progress - Rewards')
            ax1.legend()
            ax1.grid(True)
            
            # Bottom plot for car counts
            ax2.plot(car_counts, color='#e67e22', label='Avg Cars per Episode')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Number of Cars')
            ax2.set_title('Active Cars per Episode')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
        else:
            # Original single plot for rewards only
            plt.figure(figsize=(12, 6))
            
            # Plot episode rewards
            plt.plot(rewards_history, alpha=0.6, label='Episode Reward')
            
            # Plot 100-episode rolling average
            plt.plot(avg_rewards_history, label='Avg Reward (100 episodes)')
            
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Training Progress')
            plt.legend()
            plt.grid(True)
        
        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Training visualization saved to {save_path}")
        
        plt.close()
        return True
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        logger.info("Saving raw data instead...")
        
        # Save raw data as CSV if plotting fails
        if save_path:
            try:
                base_path = os.path.splitext(save_path)[0]
                with open(f"{base_path}_data.csv", 'w') as f:
                    if car_counts:
                        f.write("episode,reward,avg_reward,car_count\n")
                        for i, vals in enumerate(zip(rewards_history, avg_rewards_history, car_counts)):
                            f.write(f"{i},{vals[0]},{vals[1]},{vals[2]}\n")
                    else:
                        f.write("episode,reward,avg_reward\n")
                        for i, (r, ar) in enumerate(zip(rewards_history, avg_rewards_history)):
                            f.write(f"{i},{r},{ar}\n")
                logger.info(f"Raw data saved to {base_path}_data.csv")
                return True
            except Exception as e2:
                logger.error(f"Failed to save raw data: {e2}")
                return False
        return False

def visualize_traffic_patterns(config, save_path=None):
    """
    Visualize traffic patterns defined in the configuration.
    
    Args:
        config: Configuration dictionary containing traffic patterns
        save_path: Path to save the visualization
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Extract traffic patterns
        patterns = config.get("traffic_patterns", {})
        
        if not patterns:
            logger.warning("No traffic patterns found in configuration")
            return False
            
        # Time of day (x-axis)
        time_of_day = np.linspace(0, 24, 1440)  # 24 hours with minute resolution
        
        # Plot traffic patterns
        plt.figure(figsize=(12, 6))
        
        for pattern_name, pattern_config in patterns.items():
            # Calculate traffic intensity based on pattern
            if pattern_name == "uniform":
                # Uniform pattern has constant arrival rate
                arrival_rate = pattern_config.get("arrival_rate", 0.03)
                variability = pattern_config.get("variability", 0.01)
                
                # Add some random noise for visualization
                np.random.seed(42)  # For reproducibility
                intensity = np.ones_like(time_of_day) * arrival_rate
                intensity += np.random.normal(0, variability, size=len(time_of_day))
                
            elif pattern_name == "rush_hour":
                # Rush hour pattern has morning and evening peaks
                morning_peak = pattern_config.get("morning_peak", 0.33) * 24  # Convert to hours
                evening_peak = pattern_config.get("evening_peak", 0.71) * 24  # Convert to hours
                peak_intensity = pattern_config.get("peak_intensity", 2.0)
                base_arrival = pattern_config.get("base_arrival", 0.03)
                
                # Calculate intensity based on time of day
                morning_factor = peak_intensity * np.exp(-0.5 * ((time_of_day - morning_peak) / 1.5) ** 2)
                evening_factor = peak_intensity * np.exp(-0.5 * ((time_of_day - evening_peak) / 1.5) ** 2)
                intensity = base_arrival * (1 + morning_factor + evening_factor)
                
            elif pattern_name == "weekend":
                # Weekend pattern has one midday peak
                midday_peak = pattern_config.get("midday_peak", 0.5) * 24  # Convert to hours
                peak_intensity = pattern_config.get("peak_intensity", 1.5)
                base_arrival = pattern_config.get("base_arrival", 0.02)
                
                # Calculate intensity based on time of day
                midday_factor = peak_intensity * np.exp(-0.5 * ((time_of_day - midday_peak) / 3) ** 2)
                intensity = base_arrival * (1 + midday_factor)
                
            else:
                logger.warning(f"Unknown traffic pattern: {pattern_name}")
                continue
            
            # Plot this pattern
            plt.plot(time_of_day, intensity, label=pattern_name)
        
        # Add labels and title
        plt.xlabel('Time of Day (hours)')
        plt.ylabel('Traffic Intensity')
        plt.title('Traffic Patterns Over 24 Hours')
        plt.legend()
        plt.grid(True)
        plt.xlim(0, 24)
        plt.xticks(np.arange(0, 25, 3))
        
        # Add time labels (morning, noon, evening, night)
        plt.annotate('Morning', xy=(8, plt.ylim()[0]), xytext=(8, plt.ylim()[0] - 0.01 * (plt.ylim()[1] - plt.ylim()[0])),
                    ha='center', fontsize=10)
        plt.annotate('Noon', xy=(12, plt.ylim()[0]), xytext=(12, plt.ylim()[0] - 0.01 * (plt.ylim()[1] - plt.ylim()[0])),
                    ha='center', fontsize=10)
        plt.annotate('Evening', xy=(18, plt.ylim()[0]), xytext=(18, plt.ylim()[0] - 0.01 * (plt.ylim()[1] - plt.ylim()[0])),
                    ha='center', fontsize=10)
        plt.annotate('Night', xy=(22, plt.ylim()[0]), xytext=(22, plt.ylim()[0] - 0.01 * (plt.ylim()[1] - plt.ylim()[0])),
                    ha='center', fontsize=10)
        
        # Save the plot if path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Traffic patterns visualization saved to {save_path}")
        
        plt.close()
        return True
        
    except Exception as e:
        logger.error(f"Failed to visualize traffic patterns: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def save_visualization(env, filename="traffic_simulation.mp4", fps=30, duration=30):
    """
    Save a video visualization of the traffic simulation.
    
    Args:
        env: The traffic simulation environment
        filename: Output video file name
        fps: Frames per second
        duration: Duration of video in seconds
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import matplotlib.animation as animation
        from matplotlib import pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib.patches import Rectangle, Circle, Polygon, PathPatch
        from matplotlib.collections import PatchCollection
        import matplotlib.patches as mpatches
        import matplotlib.colors as mcolors
        
        # Make sure visualization is enabled
        old_viz_state = env.visualization
        
        # Reset environment
        env.reset()
        
        # Set up metrics tracking
        metrics_history = {
            'time': [],
            'avg_density': [],
            'waiting_time': [],
            'throughput': [],
            'avg_queue_length': []
        }
        
        # Initialize simulation metrics
        total_waiting_time = 0
        total_cars_passed = 0
        
        # Setup figure with grid layout
        fig = plt.figure(figsize=(16, 10), facecolor='#f8f9fa')
        gs = GridSpec(3, 4, figure=fig, height_ratios=[1, 5, 1])
        
        # Header area
        ax_header = fig.add_subplot(gs[0, :])
        ax_header.axis('off')
        
        # Main simulation view
        ax_main = fig.add_subplot(gs[1, :])
        
        # Statistics panels
        ax_stats1 = fig.add_subplot(gs[2, 0])
        ax_stats2 = fig.add_subplot(gs[2, 1])
        ax_stats3 = fig.add_subplot(gs[2, 2])
        ax_stats4 = fig.add_subplot(gs[2, 3])
        
        # Turn off axes for stat panels
        for ax in [ax_stats1, ax_stats2, ax_stats3, ax_stats4]:
            ax.axis('off')
        
        # Create custom colormaps for traffic density
        # Green (low) to Red (high) for traffic density
        density_cmap = mcolors.LinearSegmentedColormap.from_list(
            'density', [(0, '#00cc00'), (0.5, '#ffcc00'), (1, '#cc0000')]
        )
        
        # Blue-based colormap for North-South traffic
        ns_cmap = mcolors.LinearSegmentedColormap.from_list(
            'ns_traffic', [(0, '#e6f2ff'), (0.5, '#4d94ff'), (1, '#0047b3')]
        )
        
        # Orange-based colormap for East-West traffic
        ew_cmap = mcolors.LinearSegmentedColormap.from_list(
            'ew_traffic', [(0, '#fff2e6'), (0.5, '#ffad33'), (1, '#cc7000')]
        )
        
        # Car shapes for visualization
        def get_car_shape(x, y, direction, size=0.04):
            """Create a car shape at (x,y) pointing in the given direction."""
            if direction == 'NS':  # North-South direction (vertical)
                car_points = [
                    (x - size/2, y),            # Front middle
                    (x - size/3, y - size/2),   # Front right
                    (x + size/3, y - size/2),   # Front left
                    (x + size/2, y),            # Rear middle
                    (x + size/3, y + size/2),   # Rear left
                    (x - size/3, y + size/2),   # Rear right
                ]
            else:  # East-West direction (horizontal)
                car_points = [
                    (x, y - size/2),            # Front middle
                    (x + size/2, y - size/3),   # Front right
                    (x + size/2, y + size/3),   # Front left
                    (x, y + size/2),            # Rear middle
                    (x - size/2, y + size/3),   # Rear left
                    (x - size/2, y - size/3),   # Rear right
                ]
            return Polygon(car_points, closed=True)
        
        # Function to draw individual cars based on density
        def draw_cars(ax, i, j, direction, density, color):
            """Draw individual cars along a road segment based on density."""
            # Calculate number of cars to show (proportional to density)
            max_cars_to_show = 12  # Maximum cars to show on a road segment
            num_cars = int(density * max_cars_to_show)
            
            car_patches = []
            if direction == 'NS':
                # North-South road - cars positioned along the y-axis
                road_center_x = j + 0.5
                road_width = 0.1
                car_width = min(road_width * 0.7, 0.04)
                
                spacing = 1.0 / max(1, max_cars_to_show + 1)
                
                for k in range(num_cars):
                    car_y = i + (k + 1) * spacing
                    if car_y < i + 1:  # Ensure the car is within the road segment
                        car = get_car_shape(road_center_x, car_y, 'NS', car_width)
                        car_patches.append(car)
            else:
                # East-West road - cars positioned along the x-axis
                road_center_y = i + 0.5
                road_width = 0.1
                car_width = min(road_width * 0.7, 0.04)
                
                spacing = 1.0 / max(1, max_cars_to_show + 1)
                
                for k in range(num_cars):
                    car_x = j + (k + 1) * spacing
                    if car_x < j + 1:  # Ensure the car is within the road segment
                        car = get_car_shape(car_x, road_center_y, 'EW', car_width)
                        car_patches.append(car)
            
            # Add all cars to the plot
            if car_patches:
                car_collection = PatchCollection(car_patches, facecolor=color, edgecolor='black', 
                                                linewidth=0.5, alpha=0.85)
                ax.add_collection(car_collection)
        
        # Function to update the main visualization
        def update_main_visualization(frame):
            ax_main.clear()
            
            # Set axis properties
            ax_main.set_xlim(0, env.grid_size)
            ax_main.set_ylim(0, env.grid_size)
            ax_main.set_facecolor('#eef7ed')  # Light green background for environment
            ax_main.set_aspect('equal')
            ax_main.set_xticks([])
            ax_main.set_yticks([])
            
            # Calculate grid cell size
            cell_width = 1.0
            cell_height = 1.0
            
            # Draw background "terrain" - grass/land areas
            for i in range(env.grid_size):
                for j in range(env.grid_size):
                    # Draw grass/land background except where roads will be
                    land = Rectangle(
                        (j, i), cell_width, cell_height,
                        facecolor='#c9e5bc',  # Light green for grass/land
                        edgecolor='none',
                        alpha=0.5
                    )
                    ax_main.add_patch(land)
            
            # Draw city blocks (buildings)
            for i in range(env.grid_size):
                for j in range(env.grid_size):
                    # Draw building blocks in each cell (except where roads intersect)
                    building_margin = 0.15  # Margin from road
                    building_x = j + building_margin
                    building_y = i + building_margin
                    building_width = cell_width - 2 * building_margin
                    building_height = cell_height - 2 * building_margin
                    
                    # Randomize building colors slightly for variation
                    r = np.random.uniform(0.65, 0.75)
                    g = np.random.uniform(0.65, 0.75)
                    b = np.random.uniform(0.65, 0.75)
                    
                    building = Rectangle(
                        (building_x, building_y), building_width, building_height,
                        facecolor=(r, g, b),  # Gray with slight variation
                        edgecolor='#404040',
                        linewidth=0.5,
                        alpha=0.9
                    )
                    ax_main.add_patch(building)
            
            # Draw roads and traffic
            for i in range(env.grid_size):
                for j in range(env.grid_size):
                    idx = i * env.grid_size + j
                    
                    # Get traffic densities
                    ns_density = env.traffic_density[idx, 0]
                    ew_density = env.traffic_density[idx, 1]
                    
                    # Determine colors for roads and traffic based on density
                    ns_road_color = '#333333'  # Dark gray for base road
                    ew_road_color = '#333333'  # Dark gray for base road
                    
                    ns_traffic_color = ns_cmap(ns_density)
                    ew_traffic_color = ew_cmap(ew_density)
                    
                    # Calculate road dimensions
                    road_width = 0.1
                    ns_road_x = j + 0.5 - road_width/2
                    ew_road_y = i + 0.5 - road_width/2
                    
                    # Draw roads first (as base)
                    
                    # NS road (vertical)
                    ns_road = Rectangle(
                        (ns_road_x, i), road_width, cell_height,
                        facecolor=ns_road_color,
                        edgecolor='none',
                        alpha=0.9
                    )
                    ax_main.add_patch(ns_road)
                    
                    # EW road (horizontal)
                    ew_road = Rectangle(
                        (j, ew_road_y), cell_width, road_width,
                        facecolor=ew_road_color,
                        edgecolor='none',
                        alpha=0.9
                    )
                    ax_main.add_patch(ew_road)
                    
                    # Draw lane markings
                    def draw_lane_markings(x, y, is_vertical, length):
                        """Draw dashed lane markings on roads."""
                        if is_vertical:  # NS road
                            lane_x = x + road_width/2
                            num_dashes = int(length / 0.1)
                            for k in range(num_dashes):
                                if k % 2 == 0:  # Skip every other dash for dashed line
                                    dash_y = y + k * 0.1
                                    dash = Rectangle(
                                        (lane_x - 0.005, dash_y), 0.01, 0.05,
                                        facecolor='#ffffff',
                                        edgecolor='none',
                                        alpha=0.7
                                    )
                                    ax_main.add_patch(dash)
                        else:  # EW road
                            lane_y = y + road_width/2
                            num_dashes = int(length / 0.1)
                            for k in range(num_dashes):
                                if k % 2 == 0:  # Skip every other dash for dashed line
                                    dash_x = x + k * 0.1
                                    dash = Rectangle(
                                        (dash_x, lane_y - 0.005), 0.05, 0.01,
                                        facecolor='#ffffff',
                                        edgecolor='none',
                                        alpha=0.7
                                    )
                                    ax_main.add_patch(dash)
                    
                    # Add lane markings
                    draw_lane_markings(ns_road_x, i, True, cell_height)
                    draw_lane_markings(j, ew_road_y, False, cell_width)
                    
                    # Draw individual cars instead of just colored rectangles
                    draw_cars(ax_main, i, j, 'NS', ns_density, ns_traffic_color)
                    draw_cars(ax_main, i, j, 'EW', ew_density, ew_traffic_color)
                    
                    # Draw the intersection
                    intersection = Rectangle(
                        (ns_road_x, ew_road_y), road_width, road_width,
                        facecolor='#3a3a3a',  # Darker gray for intersection
                        edgecolor='#222222',
                        linewidth=0.5,
                        alpha=1.0
                    )
                    ax_main.add_patch(intersection)
                    
                    # Draw traffic light housing
                    light_housing_size = 0.04
                    light_housing = Rectangle(
                        (j + 0.5 - light_housing_size/2, i + 0.5 - light_housing_size/2), 
                        light_housing_size, light_housing_size,
                        facecolor='#222222',
                        edgecolor='#000000',
                        linewidth=0.5,
                        alpha=0.9
                    )
                    ax_main.add_patch(light_housing)
                    
                    # Draw traffic lights
                    light_size = 0.015
                    
                    if env.light_states[idx] == 0:  # NS Green
                        ns_color = 'green'
                        ew_color = 'red'
                    else:  # EW Green
                        ns_color = 'red'
                        ew_color = 'green'
                    
                    # NS traffic light (vertical orientation)
                    ns_light_x = j + 0.5
                    ns_light_y = i + 0.5 - light_housing_size/4
                    ns_light = Circle((ns_light_x, ns_light_y), light_size, color=ns_color, alpha=0.9)
                    ax_main.add_patch(ns_light)
                    
                    # EW traffic light (horizontal orientation)
                    ew_light_x = j + 0.5 - light_housing_size/4
                    ew_light_y = i + 0.5
                    ew_light = Circle((ew_light_x, ew_light_y), light_size, color=ew_color, alpha=0.9)
                    ax_main.add_patch(ew_light)
                    
                    # Add intersection ID
                    ax_main.text(j + 0.5, i + 0.5, f"{idx}",
                                fontsize=8, ha='center', va='center',
                                color='white', fontweight='bold')
            
            # Add scale/legend
            legend_x = 0.02
            legend_y = 0.02
            legend_width = 0.2
            legend_height = 0.08
            
            # Legend background
            legend_bg = Rectangle(
                (legend_x, legend_y), legend_width, legend_height,
                facecolor='white', alpha=0.7, transform=ax_main.transAxes
            )
            ax_main.add_patch(legend_bg)
            
            # Add legend entries
            ns_legend = Rectangle((0, 0), 1, 1, facecolor=ns_cmap(0.7), alpha=0.7)
            ew_legend = Rectangle((0, 0), 1, 1, facecolor=ew_cmap(0.7), alpha=0.7)
            ax_main.legend([ns_legend, ew_legend], ["NS Traffic", "EW Traffic"],
                          loc='lower left', bbox_to_anchor=(legend_x + 0.01, legend_y + 0.01),
                          frameon=False, fontsize=8)
            
            # Return the axis for animation
            return ax_main
        
        # Function to update the header with simulation time and status
        def update_header(frame, sim_time):
            ax_header.clear()
            ax_header.axis('off')
            
            # Calculate time metrics
            sim_seconds = sim_time
            sim_hours = int(sim_seconds / 3600)
            sim_minutes = int((sim_seconds % 3600) / 60)
            sim_seconds = int(sim_seconds % 60)
            
            # Calculate simulated time of day (0-24h) with simulation starting at 6:00 AM
            start_hour = 6  # 6 AM
            current_hour = (start_hour + sim_hours) % 24
            am_pm = "AM" if current_hour < 12 else "PM"
            display_hour = current_hour if current_hour <= 12 else current_hour - 12
            if display_hour == 0:
                display_hour = 12
            
            time_str = f"{display_hour:02d}:{sim_minutes:02d}:{sim_seconds:02d} {am_pm}"
            
            # Add time of day indicator
            time_of_day_width = 0.3
            ax_header.text(0.5, 0.7, f"Simulation Time: {time_str}", 
                         fontsize=14, fontweight='bold', ha='center', va='center')
            
            # Add day/night indicator based on time
            is_daytime = 6 <= current_hour < 18  # 6 AM to 6 PM is day
            day_night_status = "Daytime" if is_daytime else "Nighttime"
            day_night_color = '#4d94ff' if is_daytime else '#1a1a4d'
            
            day_night_indicator = Rectangle(
                (0.5 - time_of_day_width/2, 0.3), time_of_day_width, 0.2,
                facecolor=day_night_color, alpha=0.7, transform=ax_header.transAxes
            )
            ax_header.add_patch(day_night_indicator)
            
            # Add sun/moon icon
            if is_daytime:
                # Sun
                sun = Circle((0.5, 0.4), 0.03, facecolor='#ffcc00', edgecolor='#ff9900',
                           alpha=0.9, transform=ax_header.transAxes)
                ax_header.add_patch(sun)
            else:
                # Moon
                moon = Circle((0.5, 0.4), 0.03, facecolor='#f0f0f0', edgecolor='#d0d0d0',
                            alpha=0.9, transform=ax_header.transAxes)
                ax_header.add_patch(moon)
            
            # Add day/night text
            ax_header.text(0.5, 0.4, "", fontsize=8, ha='center', va='center',
                         transform=ax_header.transAxes)
            
            # Add simulation title
            ax_header.text(0.05, 0.5, "Traffic Light Management", 
                         fontsize=18, fontweight='bold', ha='left', va='center')
            
            # Add simulation parameters
            config_text = f"Grid: {env.grid_size}×{env.grid_size} | Pattern: {env.traffic_pattern}"
            ax_header.text(0.95, 0.5, config_text, 
                         fontsize=10, ha='right', va='center')
            
            return ax_header
        
        # Function to update statistics panels
        def update_stats(frame, metrics_history):
            # Calculate current metrics
            avg_density = np.mean(env.traffic_density)
            waiting_time = np.mean(env.waiting_time) if hasattr(env, 'waiting_time') else 0
            queue_lengths = np.sum(env.traffic_density > 0.5) / env.traffic_density.size
            
            # Update metrics history
            metrics_history['time'].append(frame/fps)
            metrics_history['avg_density'].append(avg_density)
            metrics_history['waiting_time'].append(waiting_time)
            metrics_history['avg_queue_length'].append(queue_lengths)
            
            # Keep only the most recent data points for plotting
            window_size = 100
            if len(metrics_history['time']) > window_size:
                for key in metrics_history:
                    metrics_history[key] = metrics_history[key][-window_size:]
            
            # Update panel 1: Overall Traffic Density
            ax_stats1.clear()
            ax_stats1.set_xlim(0, 1)
            ax_stats1.set_ylim(0, 1)
            
            # Create a density gauge
            gauge_height = 0.3
            gauge_background = Rectangle((0.1, 0.35), 0.8, gauge_height, 
                                       facecolor='#f0f0f0', edgecolor='#333333', linewidth=1)
            ax_stats1.add_patch(gauge_background)
            
            # Fill gauge based on average density
            gauge_fill = Rectangle((0.1, 0.35), 0.8 * avg_density, gauge_height, 
                                 facecolor=density_cmap(avg_density))
            ax_stats1.add_patch(gauge_fill)
            
            # Add tick marks
            for i in range(11):
                x_pos = 0.1 + i * 0.08
                ax_stats1.plot([x_pos, x_pos], [0.32, 0.35], 'k-', linewidth=1)
                if i % 2 == 0:  # Label every other tick
                    ax_stats1.text(x_pos, 0.27, f"{i*10}%", ha='center', va='top', fontsize=8)
            
            # Add panel title and value
            ax_stats1.text(0.5, 0.8, "Traffic Density", ha='center', va='center', fontsize=10, fontweight='bold')
            ax_stats1.text(0.5, 0.15, f"{avg_density*100:.1f}%", ha='center', va='center', 
                         fontsize=12, fontweight='bold', color=density_cmap(avg_density))
            
            # Update panel 2: Average Waiting Time
            ax_stats2.clear()
            ax_stats2.set_xlim(0, 1)
            ax_stats2.set_ylim(0, 1)
            
            # Plot waiting time history if we have enough data
            if len(metrics_history['time']) > 1:
                time_data = metrics_history['time']
                wait_data = metrics_history['waiting_time']
                
                # Normalize data for plotting
                wait_data_norm = np.array(wait_data) / max(1, max(wait_data))
                
                # Plot area
                ax_stats2.fill_between(
                    np.linspace(0.1, 0.9, len(time_data)), 
                    0.3, 
                    0.3 + 0.4 * wait_data_norm, 
                    color='#ff9966', alpha=0.7
                )
                
                # Add grid lines
                for y_pos in [0.3, 0.5, 0.7]:
                    ax_stats2.plot([0.1, 0.9], [y_pos, y_pos], 'k--', alpha=0.3)
            
            # Add panel title and current value
            ax_stats2.text(0.5, 0.8, "Waiting Time", ha='center', va='center', fontsize=10, fontweight='bold')
            ax_stats2.text(0.5, 0.15, f"{waiting_time:.2f}", ha='center', va='center', 
                         fontsize=12, fontweight='bold', color='#cc5200')
            
            # Update panel 3: Traffic Flow / Throughput
            ax_stats3.clear()
            ax_stats3.set_xlim(0, 1)
            ax_stats3.set_ylim(0, 1)
            
            # Calculate throughput as number of cars that passed
            if hasattr(env, 'cars_passed'):
                throughput = np.sum(env.cars_passed)
                metrics_history['throughput'].append(throughput)
            else:
                throughput = 0
                metrics_history['throughput'].append(0)
            
            # Plot throughput history if we have enough data
            if len(metrics_history['time']) > 1:
                time_data = metrics_history['time']
                throughput_data = metrics_history['throughput']
                
                # Get differences for instantaneous throughput
                if len(throughput_data) > 1:
                    instant_throughput = [throughput_data[i] - throughput_data[i-1] for i in range(1, len(throughput_data))]
                    instant_throughput = [0] + instant_throughput  # Add initial value
                    
                    # Normalize for plotting
                    max_throughput = max(1, max(instant_throughput))
                    throughput_norm = np.array(instant_throughput) / max_throughput
                    
                    # Create bar chart effect
                    bar_width = 0.8 / len(time_data)
                    for i, (t, tp) in enumerate(zip(time_data, throughput_norm)):
                        x_pos = 0.1 + i * bar_width
                        ax_stats3.add_patch(Rectangle(
                            (x_pos, 0.3), bar_width * 0.8, 0.4 * tp,
                            facecolor='#66cc99', edgecolor='none', alpha=0.7
                        ))
            
            # Add panel title and current value
            ax_stats3.text(0.5, 0.8, "Traffic Flow", ha='center', va='center', fontsize=10, fontweight='bold')
            ax_stats3.text(0.5, 0.15, f"{throughput:.0f} cars", ha='center', va='center', 
                         fontsize=12, fontweight='bold', color='#339966')
            
            # Update panel 4: Queue Lengths
            ax_stats4.clear()
            ax_stats4.set_xlim(0, 1)
            ax_stats4.set_ylim(0, 1)
            
            # Create a simple heatmap showing queue status at each intersection
            queue_heatmap_size = min(env.grid_size * 0.08, 0.3)  # Scale based on grid size
            cell_size = queue_heatmap_size / env.grid_size
            heatmap_x = 0.5 - queue_heatmap_size / 2
            heatmap_y = 0.4
            
            # Draw heatmap background
            heatmap_bg = Rectangle(
                (heatmap_x, heatmap_y), queue_heatmap_size, queue_heatmap_size,
                facecolor='#f0f0f0', edgecolor='#333333', linewidth=1
            )
            ax_stats4.add_patch(heatmap_bg)
            
            # Draw cells representing intersections
            for i in range(env.grid_size):
                for j in range(env.grid_size):
                    idx = i * env.grid_size + j
                    # Calculate average queue at this intersection
                    cell_queue = (env.traffic_density[idx, 0] + env.traffic_density[idx, 1]) / 2
                    
                    # Draw cell with color based on queue length
                    cell_x = heatmap_x + j * cell_size
                    cell_y = heatmap_y + (env.grid_size - i - 1) * cell_size  # Flip y-axis for correct orientation
                    
                    queue_cell = Rectangle(
                        (cell_x, cell_y), cell_size, cell_size,
                        facecolor=density_cmap(cell_queue), edgecolor='none'
                    )
                    ax_stats4.add_patch(queue_cell)
            
            # Add panel title and average value
            ax_stats4.text(0.5, 0.8, "Queue Status", ha='center', va='center', fontsize=10, fontweight='bold')
            ax_stats4.text(0.5, 0.15, f"Avg: {queue_lengths*100:.1f}%", ha='center', va='center', 
                         fontsize=12, fontweight='bold')
            
            return [ax_stats1, ax_stats2, ax_stats3, ax_stats4]
        
        # Main update function for animation
        def update(frame):
            # Take action every few frames to allow smoother animation
            if frame % 3 == 0:
                # Get current simulation time
                sim_time = frame / fps
                
                # Use agent if available, otherwise use intelligent/random actions
                if hasattr(env, 'recording_agent') and env.recording_agent is not None:
                    # Use trained agent
                    state = env._get_observation().flatten()
                    action = env.recording_agent.act(state, eval_mode=True)
                    _, _, _, _, info = env.step(action)
                elif frame % 15 == 0:
                    # Intelligent action: more green time for direction with higher density
                    actions = []
                    for i in range(env.num_intersections):
                        ns_density = env.traffic_density[i, 0]
                        ew_density = env.traffic_density[i, 1]
                        action = 0 if ns_density > ew_density else 1
                        actions.append(action)
                    _, _, _, _, info = env.step(actions)
                else:
                    # Random actions
                    action = np.random.randint(0, 2, size=env.num_intersections)
                    _, _, _, _, info = env.step(action)
            
            # Update all parts of the visualization
            update_main_visualization(frame)
            update_header(frame, frame / fps)
            update_stats(frame, metrics_history)
            
            return fig
        
        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=duration*fps, 
                                    interval=1000/fps, blit=False)
        
        # Save the animation
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Try to save with ffmpeg first
        try:
            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(fps=fps, metadata=dict(title='Traffic Simulation', artist='RL Agent'), bitrate=5000)
            ani.save(filename, writer=writer)
            logger.info(f"Animation saved to {filename} with FFMpegWriter")
        except Exception as e:
            logger.warning(f"FFmpeg writer failed: {e}. Trying a different approach.")
            try:
                # Try alternative save method
                ani.save(filename, fps=fps, dpi=120)
                logger.info(f"Animation saved to {filename} with alternative method")
            except Exception as e2:
                logger.error(f"Failed to save animation: {e2}")
                # Save individual frames
                frames_dir = os.path.join(os.path.dirname(filename), "frames")
                os.makedirs(frames_dir, exist_ok=True)
                
                logger.info(f"Saving individual frames to {frames_dir}...")
                for i in range(min(300, duration*fps)):  # Limit to 300 frames to avoid too many files
                    # Update figure
                    update(i)
                    # Save frame
                    frame_path = f"{frames_dir}/frame_{i:04d}.png"
                    plt.savefig(frame_path, dpi=120)
                    if i % 10 == 0:
                        logger.info(f"Saved frame {i}/{duration*fps}")
                
                logger.info(f"Frames saved. Please use an external tool to combine them into a video.")
                return False
        
        # Close figure
        plt.close(fig)
        
        # Restore original visualization state
        env.visualization = old_viz_state
        
        logger.info(f"Visualization saved to {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False