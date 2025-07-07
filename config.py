"""
Configuration file for Traffic Monitoring System
"""

# Video Processing Settings
DEFAULT_VIDEO_PATH = "input/video.mp4"
OUTPUT_VIDEO_PATH = "output/result.avi"
VIOLATION_LOG_PATH = "data/violations.csv"

# Speed Detection Settings
DEFAULT_SPEED_LIMIT = 60  # km/h
SPEED_CALCULATION_SMOOTHING = 5  # Number of frames for smoothing
MAX_REASONABLE_SPEED = 200  # km/h (for filtering unrealistic speeds)

# Vehicle Detection Settings
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck (COCO classes)
DETECTION_CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# Tracking Settings
MAX_TRACK_AGE = 50  # Maximum frames to keep inactive tracks
TRACK_INIT_FRAMES = 3  # Frames needed to confirm a track

# License Plate Recognition Settings
OCR_CONFIDENCE_THRESHOLD = 0.5
MIN_LICENSE_PLATE_LENGTH = 4
LICENSE_PLATE_LANGUAGES = ['en']

# Calibration Settings (adjust based on your camera setup)
DEFAULT_PIXELS_PER_METER = 10.0  # This should be calibrated for each camera
STANDARD_LANE_WIDTH_METERS = 3.5  # Standard lane width for calibration

# Display Settings
DISPLAY_FRAME_WIDTH = 800
DISPLAY_FRAME_HEIGHT = 600
INFO_TEXT_COLOR = (255, 255, 255)  # White
NORMAL_BBOX_COLOR = (0, 255, 0)    # Green
VIOLATION_BBOX_COLOR = (0, 0, 255)  # Red

# Performance Settings
ENABLE_GPU = True
BATCH_SIZE = 1
MEMORY_CLEANUP_INTERVAL = 100  # frames

# Logging Settings
LOG_LEVEL = "INFO"
ENABLE_CONSOLE_OUTPUT = True
PROGRESS_UPDATE_INTERVAL = 30  # frames
