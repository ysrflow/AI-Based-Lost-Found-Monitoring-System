# AI-Based Lost & Found Monitoring System

A computer-vision monitoring system that detects unattended objects from live camera streams and records event evidence for review.

Built as a final-year B.E. Computer Science and Engineering (Data Science) project.

## Overview

The system processes live RTSP camera feeds using YOLOv8 object detection and ByteTrack tracking. It identifies people and bag-like objects, monitors their movement and proximity, and flags an object when it remains stationary without a nearby person for a configured period.

When an unattended-object event is detected, the application saves video evidence and makes it available through a Flask-based dashboard.

## Key Features

- Live monitoring of two RTSP camera streams
- Object detection using YOLOv8
- Multi-object tracking with ByteTrack
- Detection of unattended and stationary bag-like objects
- Person-to-object proximity analysis
- State-based event logic for attended and unattended objects
- Buffered event recording with saved video evidence
- Flask dashboard for reviewing recorded events
- Thumbnail generation and event statistics by camera
- Video upload workflow for processing recorded footage

## How It Works

1. The application receives frames from configured RTSP cameras.
2. Frames are enhanced and passed to the YOLOv8 model.
3. The model detects people and selected bag-like object classes.
4. ByteTrack helps maintain object identity across frames.
5. The system checks whether an object is stationary and whether a person remains nearby.
6. If an object stays unattended for the configured duration, an event is created.
7. The system saves an evidence video and displays it in the Flask dashboard.

## Technology Stack

- Python
- YOLOv8
- OpenCV
- Supervision / ByteTrack
- Flask
- NumPy
- RTSP camera streams

## Project Structure

```text
ai-lost-found-monitoring/
├── code/
│   ├── main8.py              # Live camera monitoring and event detection
│   ├── app.py                # Flask dashboard
│   ├── logic/                # Object memory and state-management logic
│   ├── utils/                # Geometry utilities
│   ├── templates/            # Dashboard templates
│   ├── events/               # Generated event videos (not committed)
│   └── uploads/              # Uploaded footage (not committed)
├── docs/
│   ├── images/               # Screenshots
│   ├── research-paper.pdf    # Optional: published paper
│   └── thesis.pdf            # Optional: project thesis
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

### Prerequisites

- Python 3.10 or later
- RTSP-enabled IP camera streams, or recorded video input
- A compatible webcam/GPU setup is recommended for smoother real-time processing

### Install Dependencies

```bash
git clone https://github.com/YOUR_USERNAME/ai-lost-found-monitoring.git
cd ai-lost-found-monitoring
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

### Configure Camera Streams

Create a `.env` file using `.env.example` as a reference. Never commit the `.env` file because it contains camera connection details.

```env
CAMERA_1_URL=rtsp://username:password@camera-ip:554/stream1
CAMERA_2_URL=rtsp://username:password@camera-ip:554/stream1
```

### Run the Monitoring Service

```bash
python code/main8.py
```

### Run the Dashboard

In another terminal:

```bash
python code/app.py
```

Open the dashboard at:

```text
http://127.0.0.1:5000
```

## Detection Logic

An object is considered unattended when it meets the following conditions:

- It remains nearly stationary for the configured duration.
- No person is detected within the configured distance threshold.
- The person has remained absent for the configured memory period.
- The event has not already been triggered for that tracked object.

These thresholds can be adjusted in `main8.py` based on camera position, scene size, and expected object distance.

## Screenshots

Add project screenshots here before publishing:

- Live camera monitoring window
- Object detection and tracking output
- Flask event dashboard
- Saved event video and thumbnail

## Research and Documentation

- Research paper: add a direct link here if publicly available.
- Project thesis: add a direct link here if publicly available.

## Future Improvements

- Add email or SMS alerts for critical events
- Store event metadata in a database
- Add user authentication to the dashboard
- Support more camera streams
- Deploy the dashboard with Docker
- Add model evaluation metrics and a test dataset

## Author

Yasir Qureshi  
B.E. Computer Science and Engineering (Data Science)  
GitHub: https://github.com/ysrflow
