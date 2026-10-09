from flask import Blueprint, render_template, jsonify, request
from app.database import DatabaseManager

main_bp = Blueprint('main', __name__)
db = DatabaseManager()

@main_bp.route('/')
def dashboard():
    """Renders the main monitoring dashboard."""
    stats = db.get_summary_stats()
    return render_template('dashboard.html', stats=stats, active_page='dashboard')

@main_bp.route('/detection')
def detection():
    """Renders the real-time detection page."""
    return render_template('detection.html', active_page='detection')

@main_bp.route('/history')
def history():
    """Renders the detection history page."""
    return render_template('history.html', active_page='history')

@main_bp.route('/analytics')
def analytics():
    """Renders the analytics & charts page."""
    return render_template('analytics.html', active_page='analytics')

@main_bp.route('/settings')
def settings():
    """Renders the settings page."""
    return render_template('settings.html', active_page='settings')

@main_bp.route('/system-health')
def system_health_page():
    """Renders the system health diagnostics page."""
    return render_template('health.html', active_page='health')
