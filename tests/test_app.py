import unittest
import sys
import os
import cv2
import numpy as np

# Setup python path to include the app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.database import DatabaseManager
from app.vision.ear_calculator import EARCalculator
from app.vision.drowsiness_detector import DrowsinessDetector, DetectionState

class DriverDrowsinessSystemTests(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = cls.app.test_client()
        cls.db = DatabaseManager('tests/test_drowsiness.db')
        
    def test_01_database_initialization(self):
        """Test if the database tables are created correctly."""
        self.db.init_db()
        stats = self.db.get_summary_stats()
        self.assertIn('total_sessions', stats)
        self.assertIn('total_alerts', stats)
        self.assertIn('total_blinks', stats)
        
    def test_02_database_session_and_events(self):
        """Test database insertion logic for sessions and events."""
        import uuid
        session_id = f"test-session-{uuid.uuid4().hex[:8]}"
        self.db.start_session(session_id)
        self.db.log_blink(session_id, 0.28)
        self.db.log_drowsy_event(session_id, 0.15, 2.5, 'DANGER')
        
        events = self.db.get_recent_events(limit=10)
        self.assertTrue(len(events) > 0)
        found = any(e['session_id'] == session_id for e in events)
        self.assertTrue(found)
        
    def test_03_api_health_check(self):
        """Test API Health Check Route returns ONLINE and HTML view renders."""
        res = self.client.get('/health', headers={'Accept': 'application/json'})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['status'], 'ONLINE')
        
        # Test System Health UI route
        res_ui = self.client.get('/system-health')
        self.assertEqual(res_ui.status_code, 200)
        self.assertIn(b'System Health', res_ui.data)
        
    def test_04_api_events_export(self):
        """Test CSV and PDF export endpoints."""
        import app.routes.api_routes as api_r
        api_r.db = self.db
        
        self.db.log_drowsy_event("export-test-session", 0.1, 5.0, 'DANGER')
        
        csv_res = self.client.get('/api/export/csv')
        self.assertEqual(csv_res.status_code, 200)
        self.assertIn(b'export-test-session', csv_res.data)
        
        pdf_res = self.client.get('/api/export/pdf')
        self.assertEqual(pdf_res.status_code, 200)
        
    def test_05_frontend_routes(self):
        """Test if all front-end HTML templates render without Jinja errors."""
        routes = ['/', '/detection', '/history', '/analytics', '/settings', '/system-health']
        for route in routes:
            res = self.client.get(route)
            self.assertEqual(res.status_code, 200, f"Route {route} failed to load.")
            
    def test_06_ear_calculator(self):
        """Test eye aspect ratio math logic for both boxes and 6-point landmarks."""
        calc = EARCalculator(blink_threshold=0.23)
        img_open = np.ones((100, 100), dtype=np.uint8) * 255
        img_open[40:60, 40:60] = 0
        img_closed = np.ones((100, 100), dtype=np.uint8) * 255
        
        box = (0, 0, 100, 100)
        ear_open = calc.calculate_box_ear(box, img_open)
        ear_closed = calc.calculate_box_ear(box, img_closed)
        self.assertTrue(ear_open > ear_closed, f"EAR open {ear_open} should be > closed {ear_closed}.")
        
        # 6-point Euclidean landmark test
        # Open eye geometry: width=10, height=4
        pts_open = [(0, 0), (3, 2), (7, 2), (10, 0), (7, -2), (3, -2)]
        ear_lm_open = calc.calculate_ear(pts_open)
        # Closed eye geometry: width=10, height=0.5
        pts_closed = [(0, 0), (3, 0.25), (7, 0.25), (10, 0), (7, -0.25), (3, -0.25)]
        ear_lm_closed = calc.calculate_ear(pts_closed)
        self.assertTrue(ear_lm_open > ear_lm_closed)
        self.assertAlmostEqual(ear_lm_open, 0.40, places=2)

    def test_07_settings_and_clear_api(self):
        """Test settings retrieval/update and database clear APIs."""
        # Get settings
        res = self.client.get('/api/settings')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn('ear_threshold', data)
        
        # Update settings
        post_res = self.client.post('/api/settings', json={'ear_threshold': 0.26, 'volume': 85})
        self.assertEqual(post_res.status_code, 200)
        updated = post_res.get_json()
        self.assertEqual(updated['settings']['ear_threshold'], 0.26)
        
        # Clear data API
        clear_res = self.client.post('/api/data/clear')
        self.assertEqual(clear_res.status_code, 200)

    def test_08_state_machine_constants(self):
        """Test state machine definitions."""
        self.assertEqual(DetectionState.NORMAL, "NORMAL")
        self.assertEqual(DetectionState.EYES_CLOSED, "EYES_CLOSED")
        self.assertEqual(DetectionState.HEAD_DOWN, "HEAD_DOWN")
        self.assertEqual(DetectionState.FACE_MISSING, "FACE_MISSING")
        self.assertEqual(DetectionState.LOW_CONFIDENCE, "LOW_CONFIDENCE")
        self.assertEqual(DetectionState.CAMERA_ERROR, "CAMERA_ERROR")

if __name__ == '__main__':
    unittest.main()
