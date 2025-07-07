# Traffic Monitoring System - Simplified

A streamlined computer vision system for traffic monitoring using Python and YOLO models.

## Features

- 🚗 **Vehicle Detection**: Real-time detection using YOLOv8
- 📊 **Vehicle Counting**: Count vehicles passing through
- 🏃 **Speed Measurement**: Calculate vehicle speeds
- 🔍 **License Plate Detection**: OCR-based license plate recognition
- 📋 **Violation Logging**: Log speed violations with timestamps

## Quick Setup

1. **Install Python packages:**
```bash
pip install -r requirements.txt
```

2. **Add your video:**
   - Place any video file in the `input/` folder
   - Supported formats: MP4, AVI, MOV, MKV

3. **Run the system:**
```bash
python main.py
```

## Project Structure

```
traffic_monitoring/
├── main.py              # Main application
├── config.py            # Configuration settings
├── requirements.txt     # Dependencies
├── utils/               # Utility modules
│   ├── speed_estimator.py
│   └── tracker.py
├── input/               # Place your videos here
├── output/              # Processed videos (result.mp4)
└── data/                # Violation logs (violations.csv)
```

## Configuration

Edit these values in `main.py` or `config.py`:

- **Speed Limit**: Default 60 km/h
- **Detection Confidence**: Default 0.5
- **Pixel Calibration**: Adjust `PIXELS_PER_METER` for your camera

## Output

- **Processed Video**: `output/result.mp4` with annotations
- **Violation Log**: `data/violations.csv` with speed violations

## Dependencies

Essential packages only:
- ultralytics (YOLOv8)
- opencv-python
- numpy  
- pandas
- easyocr
- deep-sort-realtime

## Usage Tips

1. **For accurate speed measurement**: Calibrate the pixel-to-meter ratio
2. **For faster processing**: Comment out the display window in `process_video()`
3. **For different speed limits**: Change the `speed_threshold` parameter

That's it! Simple and effective traffic monitoring.
