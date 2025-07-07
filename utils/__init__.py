"""
Utility functions for Traffic Monitoring System
"""

def format_time(seconds):
    """Format seconds to HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def clean_license_plate(text):
    """Clean and format license plate text"""
    cleaned = ''.join(c for c in text if c.isalnum()).upper()
    if 4 <= len(cleaned) <= 10:
        return cleaned
    return "UNKNOWN"
