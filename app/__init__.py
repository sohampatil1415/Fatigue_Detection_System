from flask import Flask, jsonify
from app.config import Config
from app.logger import setup_logger

def create_app(config_class=Config):
    """Flask application factory."""
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize logging
    setup_logger(app)
    
    # Register Blueprints
    from app.routes.main_routes import main_bp
    from app.routes.api_routes import api_bp
    from app.routes.video_routes import video_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(video_bp)
    
    # Global HTTP error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        return jsonify({"error": "Resource Not Found", "status": 404}), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({"error": "Internal Server Error", "status": 500}), 500

    @app.route('/health')
    def health_check():
        from flask import request, render_template
        best = request.accept_mimetypes.best_match(['text/html', 'application/json'])
        if best == 'text/html' and request.accept_mimetypes[best] > request.accept_mimetypes['application/json']:
            return render_template('health.html', active_page='health')
        return jsonify({"status": "ONLINE", "message": "Driver Drowsiness System Operating Normally"})
        
    return app
