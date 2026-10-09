import os
from app import create_app

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.logger.info(f"Starting Driver Drowsiness Detection System on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)
