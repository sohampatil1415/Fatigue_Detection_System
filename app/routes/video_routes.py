import cv2
import uuid
import threading
import time
import logging
from flask import Blueprint, Response, request, jsonify
from app.vision.camera_manager import CameraManager
from app.vision.drowsiness_detector import DrowsinessDetector

logger = logging.getLogger(__name__)

video_bp = Blueprint('video', __name__)

# Global runtime state for video streaming and detection
camera = None
detector = None
stream_active = False
lock = threading.Lock()

current_settings = {
    "ear_threshold": 0.23,
    "consec_frames": 18,
    "eye_closure_duration": 0.9,
    "blink_threshold": 0.22,
    "pitch_threshold": 10.0,
    "head_down_duration": 1.0,
    "face_missing_duration": 2.0,
    "camera_index": 0,
    "resolution": "640x480",
    "alarm_enabled": True,
    "volume": 80
}

current_metrics = {
    "ear": 0.0,
    "eye_status": "AWAITING",
    "is_drowsy": False,
    "drowsy_reason": "",
    "state": "NORMAL",
    "alert_priority": "NONE",
    "head_pitch": 0.0,
    "head_pitch_baseline": 0.0,
    "is_calibrated": False,
    "total_blinks": 0,
    "drowsy_events": 0,
    "fps": 0.0
}

def parse_resolution_str(res_str):
    """Parses '640x480' string into (640, 480) integer tuple."""
    try:
        w, h = map(int, res_str.lower().split('x'))
        return (w, h)
    except Exception:
        return (640, 480)

def get_current_metrics():
    """Returns the latest metrics dict for API polling."""
    with lock:
        return dict(current_metrics)

def get_current_settings():
    """Returns the currently active runtime settings."""
    with lock:
        return dict(current_settings)

def apply_runtime_settings(new_settings):
    """Applies updated settings to camera and detector in real-time."""
    global camera, detector, current_settings
    with lock:
        current_settings.update(new_settings)

        # Update detector thresholds if detector is running
        if detector is not None:
            detector.update_settings(current_settings)

        # Update camera source or resolution if changed
        if camera is not None:
            target_res = parse_resolution_str(current_settings.get('resolution', '640x480'))
            target_idx = int(current_settings.get('camera_index', 0))

            if target_idx != camera.camera_index:
                camera.switch_camera(target_idx)
            if target_res != (camera.target_width, camera.target_height):
                camera.set_resolution(target_res[0], target_res[1])

        return dict(current_settings)

def trigger_baseline_calibration():
    """Triggers driver neutral posture calibration."""
    global detector
    with lock:
        if detector is not None:
            detector.start_calibration()
            return True
        return False

def get_vision_diagnostics():
    """Returns detailed diagnostics for system health page."""
    global camera, detector, stream_active
    with lock:
        cam_diag = camera.get_diagnostics() if camera else {
            "is_running": False, "is_synthetic": False, "camera_index": 0,
            "resolution": "N/A", "fps": 0.0, "seconds_since_last_frame": 0
        }
        return {
            "stream_active": stream_active,
            "camera": cam_diag,
            "detector_initialized": detector is not None,
            "face_mesh_ready": detector.face_mesh.mp_landmarker is not None if detector else False,
            "yunet_fallback_ready": detector.face_mesh.yunet is not None if detector else False,
            "current_state": detector.current_state if detector else "STANDBY",
            "is_calibrated": detector.is_calibrated if detector else False,
            "baseline_ear": round(detector.baseline_ear, 3) if detector else 0.32,
            "baseline_pitch": round(detector.baseline_pitch, 1) if detector else 0.0
        }

def initialize_detector(camera_index=None, resolution=None):
    """Initializes camera and drowsiness detector."""
    global camera, detector, stream_active, current_settings
    with lock:
        if camera_index is not None:
            current_settings['camera_index'] = int(camera_index)
        if resolution is not None:
            current_settings['resolution'] = resolution

        if camera is not None:
            camera.stop_camera()

        target_res = parse_resolution_str(current_settings.get('resolution', '640x480'))
        cam_idx = int(current_settings.get('camera_index', 0))

        session_id = f"session-{uuid.uuid4().hex[:8]}"
        camera = CameraManager(camera_index=cam_idx, resolution=target_res)
        detector = DrowsinessDetector(session_id)
        detector.update_settings(current_settings)
        stream_active = camera.is_running
        logger.info(f"Detector initialized: session={session_id}, stream_active={stream_active}")
    return stream_active

def generate_frames():
    """Generator that yields MJPEG frames for HTTP streaming."""
    global current_metrics, stream_active, camera, detector
    
    while stream_active:
        if camera is None or detector is None:
            time.sleep(0.05)
            continue
        
        try:
            ret, frame = camera.get_frame()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
                
            fps = camera.get_fps()
            processed_frame, metrics = detector.process(frame, fps=fps)
            
            with lock:
                current_metrics.update(metrics)
            
            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            logger.error(f"Error in video generation loop: {e}")
            time.sleep(0.05)

@video_bp.route('/video_feed')
def video_feed():
    """MJPEG video stream endpoint."""
    global stream_active
    if not stream_active or camera is None:
        initialize_detector()
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@video_bp.route('/start_camera')
def start_camera():
    """Starts the camera and detection engine."""
    global stream_active
    cam_idx = request.args.get('camera_index', None)
    res = request.args.get('resolution', None)

    # If already streaming, return reconnect status instead of re-initializing
    with lock:
        already = stream_active and camera is not None and camera.is_running

    if already:
        return jsonify({"status": "already_running"})

    success = initialize_detector(camera_index=cam_idx, resolution=res)
    return jsonify({"status": "started" if success else "failed"})

@video_bp.route('/stop_camera')
def stop_camera():
    """Stops the camera and detection engine cleanly."""
    global camera, detector, stream_active
    with lock:
        stream_active = False
        if camera:
            camera.stop_camera()
            camera = None
        detector = None
    return jsonify({"status": "stopped"})
