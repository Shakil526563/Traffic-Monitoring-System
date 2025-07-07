"""
Speed Estimator - Simple implementation for calculating vehicle speeds
"""

import numpy as np
import time
from collections import defaultdict

class SpeedEstimator:
    def __init__(self, pixel_per_meter=10.0):
        """
        Initialize speed estimator
        
        Args:
            pixel_per_meter: Calibration factor (adjust based on your camera)
        """
        self.pixel_per_meter = pixel_per_meter
        self.fps = 30
        self.track_history = defaultdict(list)
        
    def set_fps(self, fps):
        """Set video FPS for calculations"""
        self.fps = fps
    
    def calculate_speed(self, track_id, center_point):
        """
        Calculate vehicle speed based on position tracking
        
        Args:
            track_id: Unique vehicle identifier
            center_point: (x, y) center coordinates
            
        Returns:
            speed: Speed in km/h
        """
        current_time = time.time()
        
        # Store position history
        self.track_history[track_id].append({
            'position': center_point,
            'timestamp': current_time
        })
        
        # Keep only recent 5 positions for smooth calculation
        if len(self.track_history[track_id]) > 5:
            self.track_history[track_id] = self.track_history[track_id][-5:]
        
        # Need at least 2 points to calculate speed
        if len(self.track_history[track_id]) < 2:
            return 0.0
        
        # Use first and last points for calculation
        history = self.track_history[track_id]
        start_point = history[0]
        end_point = history[-1]
        
        # Calculate distance in pixels
        dx = end_point['position'][0] - start_point['position'][0]
        dy = end_point['position'][1] - start_point['position'][1]
        distance_pixels = np.sqrt(dx**2 + dy**2)
        
        # Calculate time difference
        time_diff = end_point['timestamp'] - start_point['timestamp']
        
        if time_diff > 0:
            # Convert to real-world speed
            distance_meters = distance_pixels / self.pixel_per_meter
            speed_mps = distance_meters / time_diff  # meters per second
            speed_kmh = speed_mps * 3.6  # convert to km/h
            
            # Apply reasonable bounds (0-200 km/h)
            return max(0, min(200, speed_kmh))
        
        return 0.0
    
    def set_calibration(self, pixel_per_meter):
        """Update calibration factor"""
        self.pixel_per_meter = pixel_per_meter
