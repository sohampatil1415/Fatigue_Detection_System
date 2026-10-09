import io
import time
import os
import psutil
from flask import Blueprint, jsonify, request, Response, send_file
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from app.database import DatabaseManager

api_bp = Blueprint('api', __name__, url_prefix='/api')
db = DatabaseManager()

START_TIME = time.time()

@api_bp.route('/stats')
def get_stats():
    """Returns aggregate system statistics."""
    stats = db.get_summary_stats()
    return jsonify(stats)

@api_bp.route('/analytics')
def get_analytics():
    """Returns time-bucketed analytics data for charts."""
    period = request.args.get('period', 'daily')
    data = db.get_analytics_data(period=period)
    return jsonify(data)

@api_bp.route('/cameras')
def discover_cameras():
    """Probes available camera indices (0-4) and returns which ones open."""
    import cv2
    available = []
    for idx in range(5):
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    available.append({"index": idx, "label": f"Camera {idx}"})
            else:
                cap.release()
        except Exception:
            pass
    if not available:
        available.append({"index": 0, "label": "Camera 0 (Default)"})
    return jsonify({"cameras": available})

@api_bp.route('/test-camera', methods=['POST'])
def test_camera():
    """Opens requested camera index, grabs one frame, returns diagnostic result."""
    import cv2
    data = request.get_json() or {}
    idx = int(data.get('camera_index', 0))
    result = {"camera_index": idx, "success": False, "message": "", "resolution": ""}
    try:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                result["success"] = True
                result["message"] = f"Camera {idx} opened and produced a valid frame."
                result["resolution"] = f"{w}x{h}"
            else:
                result["message"] = f"Camera {idx} opened but failed to capture a frame."
        else:
            result["message"] = f"Camera {idx} could not be opened — device unavailable or in use."
    except Exception as e:
        result["message"] = f"Camera test exception: {str(e)}"
    return jsonify(result)

@api_bp.route('/events')
def get_events():
    """Returns recent drowsiness events with optional limit."""
    limit = request.args.get('limit', 50, type=int)
    events = db.get_recent_events(limit=limit)
    return jsonify(events)

@api_bp.route('/events/search')
def search_events():
    """Search events by alert level or session."""
    alert_level = request.args.get('alert_level', '')
    session_id = request.args.get('session_id', '')
    limit = request.args.get('limit', 50, type=int)
    events = db.get_recent_events(limit=limit)
    if alert_level:
        events = [e for e in events if e.get('alert_level', '').upper() == alert_level.upper()]
    if session_id:
        events = [e for e in events if session_id.lower() in e.get('session_id', '').lower()]
    return jsonify(events)

@api_bp.route('/events/<int:event_id>', methods=['DELETE'])
def delete_event(event_id):
    """Deletes a specific drowsiness event by ID."""
    try:
        db.delete_event(event_id)
        return jsonify({"status": "deleted", "id": event_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/export/csv')
def export_csv():
    """Exports drowsiness events as CSV download."""
    csv_data = db.export_events_csv()
    return Response(
        csv_data,
        mimetype='text/csv',
        headers={"Content-Disposition": "attachment;filename=drowsiness_events.csv"}
    )

@api_bp.route('/export/pdf')
def export_pdf():
    """Exports drowsiness events as a PDF report."""
    events = db.get_recent_events(limit=500)
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    
    styles = getSampleStyleSheet()
    elements.append(Paragraph("DrowsiGuard AI - Detection Report", styles['Title']))
    elements.append(Spacer(1, 12))
    
    data = [['ID', 'Session', 'Timestamp', 'EAR', 'Duration', 'Level']]
    for e in events:
        data.append([
            str(e.get('id', '')),
            str(e.get('session_id', ''))[:8],
            str(e.get('timestamp', '')),
            f"{float(e.get('ear_val', 0)):.3f}",
            f"{float(e.get('duration_seconds', 0)):.2f}s",
            str(e.get('alert_level', ''))
        ])
    
    if len(data) == 1:
        data.append(['—', 'No events logged', '—', '—', '—', '—'])

    t = Table(data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e2433')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 10),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#f8f9fa')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
    ]))
    
    elements.append(t)
    doc.build(elements)
    
    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name='drowsiness_report.pdf',
        mimetype='application/pdf'
    )

@api_bp.route('/metrics')
def live_metrics():
    """Returns current live detection metrics (polled by frontend)."""
    from app.routes.video_routes import get_current_metrics
    metrics = get_current_metrics()
    return jsonify(metrics)

@api_bp.route('/settings', methods=['GET', 'POST'])
def manage_settings():
    """Retrieves or saves application settings."""
    from app.routes.video_routes import get_current_settings, apply_runtime_settings
    if request.method == 'POST':
        data = request.get_json() or {}
        updated = apply_runtime_settings(data)
        return jsonify({"status": "success", "settings": updated})
    else:
        return jsonify(get_current_settings())

@api_bp.route('/calibrate', methods=['POST'])
def calibrate():
    """Triggers driver neutral posture calibration."""
    from app.routes.video_routes import trigger_baseline_calibration
    success = trigger_baseline_calibration()
    return jsonify({"status": "calibrating" if success else "failed", "message": "Baseline calibration initiated."})

@api_bp.route('/data/clear', methods=['POST'])
def clear_data():
    """Deletes all session and event history from database."""
    try:
        db.clear_all_data()
        return jsonify({"status": "cleared", "message": "All detection records have been cleared."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/health')
def system_health():
    """Provides complete system diagnostics and telemetry."""
    from app.routes.video_routes import get_vision_diagnostics
    uptime_secs = int(time.time() - START_TIME)
    
    # Process & Hardware metrics
    cpu_percent = psutil.cpu_percent(interval=None)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(os.path.abspath(os.sep))
    process = psutil.Process()
    proc_mem = process.memory_info().rss / (1024 * 1024)

    vision_diag = get_vision_diagnostics()
    db_diag = db.get_health_stats()

    overall_status = "HEALTHY"
    if db_diag.get("status") != "HEALTHY":
        overall_status = "DEGRADED"

    return jsonify({
        "status": overall_status,
        "uptime_seconds": uptime_secs,
        "uptime_formatted": f"{uptime_secs // 3600}h {(uptime_secs % 3600) // 60}m {uptime_secs % 60}s",
        "system": {
            "cpu_usage_percent": cpu_percent,
            "ram_used_percent": memory.percent,
            "ram_available_mb": round(memory.available / (1024 * 1024), 1),
            "disk_free_gb": round(disk.free / (1024 * 1024 * 1024), 2),
            "process_memory_mb": round(proc_mem, 1)
        },
        "database": db_diag,
        "vision": vision_diag,
        "audio": {
            "status": "READY",
            "device": "Pygame Mixer / Web Audio API"
        }
    })
