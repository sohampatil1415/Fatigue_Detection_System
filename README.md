# Fatigue_Detection_System

![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-5.0-green.svg)
![License](https://img.shields.io/badge/license-MIT-purple.svg)

A production-ready, enterprise-grade AI driver monitoring system built with Python, OpenCV, and Flask. This system actively monitors a video feed to detect signs of driver fatigue in real-time, offering instant audio alerts, comprehensive analytics, and historical reporting.

---

## 🌟 Key Features

* **Real-Time AI Detection Engine:** Custom-built YCrCb skin-contour tracking and Eye Aspect Ratio (EAR) metric calculation with continuous frame heuristics.
* **Non-Blocking Multithreaded IO:** Dedicated background threads handle camera feed buffering for latency-free OpenCV operations.
* **Professional Glassmorphic Dashboard:** A completely responsive UI built with Vanilla CSS and FontAwesome, featuring dark-theme aesthetics.
* **Comprehensive Analytics:** Integrated Chart.js visualizations for daily, weekly, and monthly alert trends.
* **Data Management & Export:** Integrated SQLite database tracking blink rates and drowsiness events, with single-click CSV and PDF report generation.
* **Persistent Hardware Controls:** Manage EAR thresholds, frame counts, camera sources, and volume levels dynamically with `localStorage` state persistence.

---

## 🏗️ Project Architecture

```
AI_Driver_Drowsiness_Detection/
│
├── app/
│   ├── routes/                # Blueprint Controllers
│   │   ├── main_routes.py     # HTML Page Renderers
│   │   ├── api_routes.py      # REST APIs & PDF/CSV Exports
│   │   └── video_routes.py    # MJPEG Streaming Engine
│   │
│   ├── static/
│   │   ├── css/style.css      # Core Design System
│   │   └── sounds/alarm.wav   # Pygame Audio Alert
│   │
│   ├── templates/             # Jinja2 Views
│   │   ├── base.html          # Layout Scaffolding
│   │   ├── dashboard.html     # Live Command Center
│   │   ├── detection.html     # Camera HUD Interface
│   │   ├── history.html       # Event Logs Table
│   │   ├── analytics.html     # Chart.js Visualizations
│   │   └── settings.html      # Hardware Configurations
│   │
│   ├── vision/                # AI Detection Core
│   │   ├── camera_manager.py  # Threaded Frame Grabber
│   │   ├── ear_calculator.py  # Euclidean Distance Math
│   │   ├── face_mesh.py       # YCrCb Skin Contour Logic
│   │   └── drowsiness_detector.py # State Machine & HUD Overlay
│   │
│   ├── config.py              # System Configuration variables
│   ├── database.py            # SQLite ORM via Python sqlite3/pandas
│   ├── logger.py              # System diagnostics
│   └── __init__.py            # Flask App Factory
│
├── tests/
│   └── test_app.py            # Automated Regression Suite
│
├── run.py                     # WSGI Application Entry Point
└── requirements.txt           # Python Dependency Manifest
```

---

## 🚀 Installation Guide

### Prerequisites
* Python 3.9 or higher
* A working USB or integrated Webcam
* Visual Studio C++ Build Tools (Windows only, for specific OpenCV dependencies)

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/drowsiguard-ai.git
   cd drowsiguard-ai
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Application**
   ```bash
   python run.py
   ```

5. **Access the Dashboard**
   Open your browser and navigate to: `http://localhost:5000`

---

## 📖 User Manual & API Documentation

### Starting a Session
1. Navigate to the **Live Detection** tab in the sidebar.
2. Click the green **Start** button. The system will initialize the camera.
3. The HUD will overlay your face with tracking boxes. Keep your eyes visible to the camera.

### Understanding Alerts
* If the EAR (Eye Aspect Ratio) drops below the threshold (default `0.25`) for `20` consecutive frames, the system triggers a **DANGER** alert.
* An audible alarm will sound through your speakers (adjustable via the Settings page).
* The event is immediately logged to the SQLite database.

### REST API Documentation
* `GET /api/stats` - Returns aggregate JSON system statistics (blinks, sessions, alerts).
* `GET /api/events?limit=50` - Returns a JSON array of recent detection anomalies.
* `GET /api/export/csv` - Downloads a raw CSV of all historical logs.
* `GET /api/export/pdf` - Generates and downloads a formatted PDF incidence report via ReportLab.
* `GET /video_feed` - Multipart MJPEG stream consuming the OpenCV processed frames.

---

## 🔮 Future Scope
* **Infrared Camera Support:** Integration with IR sensors for night-driving accuracy without dashboard glare.
* **Advanced Deep Learning:** Migrating from mathematical YCrCb contours to lightweight MobileNet/YOLO architectures for extreme edge-case robustness.
* **Cloud Telemetry:** Implementing MQTT protocols to stream alerts back to central fleet management dashboards for commercial trucking companies.

---

## 📚 References
* [OpenCV Documentation](https://docs.opencv.org/)
* [Flask Web Development](https://flask.palletsprojects.com/)
* [ReportLab PDF Generation](https://www.reportlab.com/)
* Soukupová, T. and Čech, J. (2016). *Real-Time Eye Blink Detection using Facial Landmarks*. 
