"""
Vehicle Tracker - Simple implementation for counting and tracking vehicles
"""

class VehicleTracker:
    def __init__(self):
        """Initialize vehicle tracker"""
        self.tracked_vehicles = set()
        self.vehicle_positions = {}
        
    def update_track(self, track_id, bbox):
        """
        Update vehicle tracking information
        
        Args:
            track_id: Unique vehicle identifier
            bbox: Bounding box coordinates (x1, y1, x2, y2)
        """
        # Add vehicle to tracked set
        self.tracked_vehicles.add(track_id)
        
        # Store current position
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        self.vehicle_positions[track_id] = (center_x, center_y)
    
    def get_vehicle_count(self):
        """Get total number of unique vehicles tracked"""
        return len(self.tracked_vehicles)
    
    def get_vehicle_position(self, track_id):
        """Get current position of a vehicle"""
        return self.vehicle_positions.get(track_id, (0, 0))
    
    def is_new_vehicle(self, track_id):
        """Check if this is a newly detected vehicle"""
        return track_id not in self.tracked_vehicles
