
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os

try:
    from ultralytics import YOLO
    from deep_sort_realtime.deepsort_tracker import DeepSort
    import easyocr
except ImportError as e:
    print(f"❌ Missing package: {e}")
    print("Install with: pip install ultralytics deep-sort-realtime easyocr")
    exit(1)

from utils.speed_estimator import SpeedEstimator
from utils.tracker import VehicleTracker

class TrafficMonitor:
    def __init__(self, video_path, speed_threshold=60, debug_mode=False):
        """Initialize the traffic monitoring system"""
        self.video_path = video_path
        self.speed_threshold = speed_threshold
        self.debug_mode = debug_mode
        
        # Initialize models
        print("Loading models...")
        model_path = 'models/yolov8n.pt'
        if not os.path.exists(model_path):
            model_path = 'yolov8n.pt'
            print("⚠️ Model not found in 'models/' folder, using default location")
        else:
            print(f"✅ Loading model from: {model_path}")
        
        self.vehicle_model = YOLO(model_path)
        self.tracker = DeepSort(max_age=50, n_init=3)
        self.ocr_reader = easyocr.Reader(['en'])
        
        # Initialize components
        self.speed_estimator = SpeedEstimator()
        self.vehicle_tracker = VehicleTracker()
        
      
        self.license_plate_cache = {}
        self.plate_detection_interval = 90  
        self.frame_count = 0
        

        self.vehicle_data = {}
        self.data_batch = []
        self.batch_size = 10  # Write to CSV every 10 entries
        
        # Create directories
        os.makedirs('output', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        
        # Initialize data log
        self.init_data_log()
        
    def init_data_log(self):
        """Initialize data CSV file for all vehicles"""
        if not os.path.exists('data/vehicle_data.csv'):
            df = pd.DataFrame(columns=['timestamp', 'track_id', 'license_plate', 'speed_kmh', 'is_violation'])
            df.to_csv('data/vehicle_data.csv', index=False)
        
        # Also keep violations file for backward compatibility
        if not os.path.exists('data/violations.csv'):
            df = pd.DataFrame(columns=['timestamp', 'track_id', 'license_plate', 'speed_kmh'])
            df.to_csv('data/violations.csv', index=False)
    
    def detect_vehicles(self, frame):
        """Detect vehicles in frame using YOLO"""
        results = self.vehicle_model(frame, classes=[2, 3, 5, 7], verbose=False)
        
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    if conf > 0.5:
                        # Convert from xyxy to ltwh format for DeepSort
                        left, top, width, height = x1, y1, x2 - x1, y2 - y1
                        detections.append(([left, top, width, height], conf, 'vehicle'))
        
        return detections
    
    def detect_license_plate(self, vehicle_crop):
        """Fast license plate detection with minimal processing"""
        try:
            h, w = vehicle_crop.shape[:2]
            
            # Skip very small crops
            if h < 25 or w < 60:
                return "UNKNOWN"
            
    
            plate_region = vehicle_crop[int(h*0.6):, :]
            
      
            gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
            
        
            gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=20)
            
            # Single OCR attempt with character whitelist
            results = self.ocr_reader.readtext(gray, 
                                             allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                                             width_ths=0.7,
                                             height_ths=0.7)
            
            # Find best result quickly
            for (bbox, text, conf) in results:
                if conf > 0.3:  # Higher confidence threshold
                    cleaned_text = self.clean_license_plate_text(text)
                    if cleaned_text and self.is_valid_license_plate(cleaned_text):
                        return cleaned_text
            
            return "UNKNOWN"
            
        except Exception:
            return "UNKNOWN"
    
    def clean_license_plate_text(self, text):
        """Fast license plate text cleaning"""
        if not text:
            return None
            
        # Quick cleanup
        text = text.upper().strip()
        cleaned = ''.join(c for c in text if c.isalnum())
        
        # Filter by length quickly
        if len(cleaned) < 4 or len(cleaned) > 8:
            return None
            
        return cleaned
    
    def get_plate_pattern_score(self, text):
        """Score license plate based on common patterns"""
        if not text:
            return 0
        
        score = 1.0
        
        # Prefer length 5-7
        if 5 <= len(text) <= 7:
            score *= 1.5
        elif len(text) == 4 or len(text) == 8:
            score *= 1.2
        
        has_letters = any(c.isalpha() for c in text)
        has_numbers = any(c.isdigit() for c in text)
        
        if has_letters and has_numbers:
            score *= 1.3
        
        letter_start = text[:3].isalpha() if len(text) >= 3 else False
        if letter_start:
            score *= 1.2
        
        return score
    
    def is_valid_license_plate(self, text):
        """Fast license plate validation"""
        if not text or len(text) < 4 or len(text) > 8:
            return False
        
        # Quick checks
        has_letters = any(c.isalpha() for c in text)
        has_numbers = any(c.isdigit() for c in text)
        
        # Must have both letters and numbers
        return has_letters and has_numbers

    def log_vehicle_data(self, track_id, license_plate, speed, is_violation):
        """Efficient batch logging of vehicle data"""
        vehicle_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'track_id': track_id,
            'license_plate': license_plate,
            'speed_kmh': round(speed, 1),
            'is_violation': is_violation
        }
        
        # Add to batch
        self.data_batch.append(vehicle_data)
        
        # Write batch when full
        if len(self.data_batch) >= self.batch_size:
            self.flush_data_batch()
        
        # Log violations immediately for alerts
        if is_violation:
            violation_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'track_id': track_id,
                'license_plate': license_plate,
                'speed_kmh': round(speed, 1)
            }
            df_violation = pd.DataFrame([violation_data])
            df_violation.to_csv('data/violations.csv', mode='a', header=False, index=False)
            print(f"🚨 VIOLATION: Track {track_id}, Speed: {speed:.1f} km/h, Plate: {license_plate}")
    
    def flush_data_batch(self):
        """Write batched data to CSV"""
        if self.data_batch:
            df = pd.DataFrame(self.data_batch)
            df.to_csv('data/vehicle_data.csv', mode='a', header=False, index=False)
            self.data_batch = []

    def log_violation(self, track_id, license_plate, speed):
        """Log speed violation - kept for backward compatibility"""
        self.log_vehicle_data(track_id, license_plate, speed, True)
    
    def draw_info(self, frame, track_id, bbox, speed, license_plate="", is_violation=False):
        """Draw vehicle information on frame with enhanced license plate display"""
        x1, y1, x2, y2 = map(int, bbox)
        color = (0, 0, 255) if is_violation else (0, 255, 0)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw main info text
        info = f"ID:{track_id} Speed:{speed:.1f}km/h"
        cv2.putText(frame, info, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if license_plate and license_plate != "UNKNOWN":
            plate_text = f"Plate: {license_plate}"
            (text_width, text_height), _ = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            plate_y = y1 - 35
            plate_x = x1
            
            cv2.rectangle(frame, (plate_x - 5, plate_y - text_height - 5), 
                         (plate_x + text_width + 5, plate_y + 5), (0, 0, 0), -1)
            
            # Draw the license plate text in bright color
            plate_color = (0, 255, 255) if is_violation else (255, 255, 0)  # Yellow/Cyan
            cv2.putText(frame, plate_text, (plate_x, plate_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, plate_color, 2)
        
        if is_violation:
            cv2.putText(frame, "SPEEDING!", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    def process_video(self, show_live=True):
        """Main video processing function with live preview option"""
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"❌ Cannot open video: {self.video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Processing: {width}x{height}, {fps}FPS, {total_frames} frames")
        if show_live:
            print("🖥️  Live preview enabled - Press 'q' to quit, 'p' to pause/resume")
        
        # Setup output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output/result.mp4', fourcc, fps, (width, height))
        
        self.speed_estimator.set_fps(fps)
        
        paused = False
        if show_live:
            cv2.namedWindow('Traffic Monitor - Live Feed', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Traffic Monitor - Live Feed', 1280, 720)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            # Detect and track vehicles
            detections = self.detect_vehicles(frame)
            tracks = self.tracker.update_tracks(detections, frame=frame)
            
            # Process each tracked vehicle
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                track_id = track.track_id
                bbox = track.to_ltwh()
                x1, y1, w, h = bbox
                x2, y2 = x1 + w, y1 + h
                center = (x1 + w/2, y1 + h/2)
                
                # Update tracking and calculate speed
                self.vehicle_tracker.update_track(track_id, (x1, y1, x2, y2))
                speed = self.speed_estimator.calculate_speed(track_id, center)
                
                license_plate = self.license_plate_cache.get(track_id, "UNKNOWN")
                
                should_detect_plate = (
                    license_plate == "UNKNOWN" and 
                    self.frame_count % self.plate_detection_interval == 0 and
                    speed > 5  # Only for vehicles with some speed
                )
                
                if should_detect_plate:
                    crop_y1 = max(0, int(y2 - h * 0.4))
                    crop_y2 = min(height, int(y2))
                    crop_x1 = max(0, int(x1))
                    crop_x2 = min(width, int(x2))
                    
                    if crop_y2 > crop_y1 and crop_x2 > crop_x1:
                        vehicle_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                        if vehicle_crop.size > 0:
                            detected_plate = self.detect_license_plate(vehicle_crop)
                            if detected_plate != "UNKNOWN":
                                license_plate = detected_plate
                                self.license_plate_cache[track_id] = license_plate
                
                # Check for speed violation
                is_violation = speed > self.speed_threshold
                
                if speed > 0.5:  # Filter out stationary vehicles
                    vehicle_key = f"{track_id}_{int(speed/5)*5}"  # Group by 5 km/h intervals
                    if vehicle_key not in self.vehicle_data or self.frame_count % 60 == 0:
                        self.vehicle_data[vehicle_key] = self.frame_count
                        self.log_vehicle_data(track_id, license_plate, speed, is_violation)
                
                # Draw vehicle info
                self.draw_info(frame, track_id, (x1, y1, x2, y2), speed, license_plate, is_violation)
            
            # Draw statistics
            cv2.putText(frame, f"Frame: {self.frame_count}/{total_frames}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Vehicles: {self.vehicle_tracker.get_vehicle_count()}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Speed Limit: {self.speed_threshold} km/h", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if show_live:
                # Add pause/resume indicator
                if paused:
                    cv2.putText(frame, "PAUSED - Press 'p' to resume", (10, height - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, "LIVE - Press 'q' to quit, 'p' to pause", (10, height - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Display the frame
                cv2.imshow('Traffic Monitor - Live Feed', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n⏹️  Live preview stopped by user")
                    break
                elif key == ord('p'):
                    paused = not paused
                    print(f"⏸️  {'Paused' if paused else 'Resumed'}")
                
                # Skip processing if paused
                if paused:
                    continue
            
            # Save frame
            out.write(frame)
            
            # Progress indicator
            if self.frame_count % 100 == 0:
                progress = (self.frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}%")
        
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Flush any remaining data
        self.flush_data_batch()
        
        print(f"\n Complete! Processed {self.frame_count} frames")
        print(f"Total vehicles: {self.vehicle_tracker.get_vehicle_count()}")
        print(f" Output: output/result.mp4")
        print(f" All vehicle data: data/vehicle_data.csv")
        print(f" Violations only: data/violations.csv")

def main():
    """Main function"""
    print("Traffic Monitoring System")
    print("=" * 40)
    
    # Find video file
    input_dir = "input"
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = []
    
    if os.path.exists(input_dir):
        video_files = [f for f in os.listdir(input_dir) 
                      if f.lower().endswith(video_extensions)]
    
    if not video_files:
        print(" No video file found in 'input/' folder")
        print("Please add a video file (MP4, AVI, MOV, MKV)")
        return
    
    video_path = os.path.join(input_dir, video_files[0])
    print(f"Processing: {video_files[0]}")
    
    # Process video with 60 km/h speed limit, debug mode disabled for speed, and live preview
    monitor = TrafficMonitor(video_path, speed_threshold=60, debug_mode=False)
    monitor.process_video(show_live=True)

if __name__ == "__main__":
    main()
