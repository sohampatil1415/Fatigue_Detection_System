import os
import sqlite3
import pandas as pd
from datetime import datetime
from app.config import Config

class DatabaseManager:
    """SQLite Database Data Access Layer with Pandas Export capabilities."""
    
    def __init__(self, db_path=None):
        self.db_path = db_path or Config.DATABASE_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    def get_connection(self):
        """Creates sqlite3 database connection with dictionary rows."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Initializes database schema and indexes."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Sessions Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    end_time DATETIME,
                    total_blinks INTEGER DEFAULT 0,
                    total_drowsy_events INTEGER DEFAULT 0,
                    avg_ear REAL DEFAULT 0.0,
                    status TEXT DEFAULT 'ACTIVE'
                )
            ''')

            # Drowsiness Events Log Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS drowsiness_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    ear_val REAL,
                    duration_seconds REAL,
                    alert_level TEXT,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
            ''')

            # Blink Logs Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS blink_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    ear_val REAL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
            ''')
            
            conn.commit()

    def start_session(self, session_id):
        """Registers a new driver detection session."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO sessions (session_id, start_time, status) VALUES (?, ?, 'ACTIVE')",
                (session_id, datetime.now().isoformat())
            )
            conn.commit()

    def end_session(self, session_id, total_blinks, total_drowsy_events, avg_ear):
        """Closes an active session with aggregated metrics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE sessions 
                SET end_time = ?, total_blinks = ?, total_drowsy_events = ?, avg_ear = ?, status = 'COMPLETED'
                WHERE session_id = ?
            ''', (datetime.now().isoformat(), total_blinks, total_drowsy_events, avg_ear, session_id))
            conn.commit()

    def log_drowsy_event(self, session_id, ear_val, duration_seconds, alert_level='WARNING'):
        """Logs a drowsiness detection incident."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO drowsiness_events (session_id, timestamp, ear_val, duration_seconds, alert_level)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, datetime.now().isoformat(), ear_val, duration_seconds, alert_level))
            
            # Increment event counter on active session
            cursor.execute('''
                UPDATE sessions SET total_drowsy_events = total_drowsy_events + 1 WHERE session_id = ?
            ''', (session_id,))
            conn.commit()

    def log_blink(self, session_id, ear_val):
        """Logs an eye blink event."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO blink_logs (session_id, timestamp, ear_val) VALUES (?, ?, ?)
            ''', (session_id, datetime.now().isoformat(), ear_val))
            
            cursor.execute('''
                UPDATE sessions SET total_blinks = total_blinks + 1 WHERE session_id = ?
            ''', (session_id,))
            conn.commit()

    def get_recent_events(self, limit=50):
        """Retrieves recent drowsiness events for dashboard reporting."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, session_id, timestamp, ear_val, duration_seconds, alert_level
                FROM drowsiness_events ORDER BY id DESC LIMIT ?
            ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_summary_stats(self):
        """Computes system-wide aggregate statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) as total_sessions FROM sessions')
            total_sessions = cursor.fetchone()['total_sessions']
            
            cursor.execute('SELECT COUNT(*) as total_alerts FROM drowsiness_events')
            total_alerts = cursor.fetchone()['total_alerts']
            
            cursor.execute('SELECT COUNT(*) as total_blinks FROM blink_logs')
            total_blinks = cursor.fetchone()['total_blinks']
            
            return {
                "total_sessions": total_sessions,
                "total_alerts": total_alerts,
                "total_blinks": total_blinks
            }

    def delete_event(self, event_id):
        """Deletes a drowsiness event by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM drowsiness_events WHERE id = ?', (event_id,))
            conn.commit()

    def get_sessions(self, limit=20):
        """Retrieves recent sessions."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT session_id, start_time, end_time, total_blinks, total_drowsy_events, avg_ear, status
                FROM sessions ORDER BY start_time DESC LIMIT ?
            ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def export_events_csv(self):
        """Exports drowsiness events table to CSV string via Pandas."""
        with self.get_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM drowsiness_events ORDER BY timestamp DESC", conn)
            return df.to_csv(index=False)

    def clear_all_data(self):
        """Wipes all drowsiness events and blink logs from the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM drowsiness_events")
            cursor.execute("DELETE FROM blink_logs")
            cursor.execute("DELETE FROM sessions")
            conn.commit()
            return True

    def get_health_stats(self):
        """Returns database health metrics for system health monitoring."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA integrity_check")
                integrity = cursor.fetchone()[0]
                size_bytes = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
                return {
                    "status": "HEALTHY" if integrity == "ok" else "DEGRADED",
                    "integrity": integrity,
                    "file_size_kb": round(size_bytes / 1024, 2),
                    "db_path": self.db_path
                }
        except Exception as e:
            return {
                "status": "ERROR",
                "integrity": str(e),
                "file_size_kb": 0,
                "db_path": self.db_path
            }

    def get_analytics_data(self, period='daily'):
        """Returns time-bucketed analytics data for charts.
        
        period: 'daily' (today by hour), 'weekly' (last 7 days), 'monthly' (last 4 weeks)
        """
        from datetime import timedelta
        now = datetime.now()

        with self.get_connection() as conn:
            cursor = conn.cursor()

            if period == 'daily':
                labels = [f"{h}:00" for h in range(0, 24, 3)] + ['23:59']
                bucket_count = 9
                start = now.replace(hour=0, minute=0, second=0, microsecond=0)
                bucket_seconds = 3 * 3600  # 3 hours per bucket
            elif period == 'weekly':
                days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                today_weekday = now.weekday()
                start = (now - timedelta(days=today_weekday)).replace(hour=0, minute=0, second=0, microsecond=0)
                labels = days
                bucket_count = 7
                bucket_seconds = 86400  # 1 day per bucket
            else:  # monthly
                labels = ['Week 1', 'Week 2', 'Week 3', 'Week 4']
                start = (now - timedelta(days=28)).replace(hour=0, minute=0, second=0, microsecond=0)
                bucket_count = 4
                bucket_seconds = 7 * 86400  # 1 week per bucket

            start_iso = start.isoformat()

            # Fetch drowsiness events in range
            cursor.execute(
                "SELECT timestamp, ear_val, alert_level FROM drowsiness_events WHERE timestamp >= ? ORDER BY timestamp",
                (start_iso,)
            )
            events = cursor.fetchall()

            alert_counts = [0] * bucket_count
            danger_count = 0
            warning_count = 0
            critical_count = 0
            ear_values = []

            for row in events:
                try:
                    ts = datetime.fromisoformat(row['timestamp'])
                    delta = (ts - start).total_seconds()
                    bucket = min(int(delta / bucket_seconds), bucket_count - 1)
                    if bucket >= 0:
                        alert_counts[bucket] += 1
                except (ValueError, TypeError):
                    pass

                level = (row['alert_level'] or '').upper()
                if level == 'DANGER':
                    danger_count += 1
                elif level == 'CRITICAL':
                    critical_count += 1
                else:
                    warning_count += 1

                try:
                    ear_values.append(float(row['ear_val']))
                except (ValueError, TypeError):
                    pass

            # Fetch blink data in range
            cursor.execute(
                "SELECT timestamp FROM blink_logs WHERE timestamp >= ? ORDER BY timestamp",
                (start_iso,)
            )
            blinks = cursor.fetchall()

            blink_counts = [0] * bucket_count
            for row in blinks:
                try:
                    ts = datetime.fromisoformat(row['timestamp'])
                    delta = (ts - start).total_seconds()
                    bucket = min(int(delta / bucket_seconds), bucket_count - 1)
                    if bucket >= 0:
                        blink_counts[bucket] += 1
                except (ValueError, TypeError):
                    pass

            # EAR histogram
            ear_buckets = [0, 0, 0, 0, 0]
            ear_labels = ['0.00–0.15', '0.15–0.20', '0.20–0.25', '0.25–0.30', '0.30+']
            for v in ear_values:
                if v < 0.15:
                    ear_buckets[0] += 1
                elif v < 0.20:
                    ear_buckets[1] += 1
                elif v < 0.25:
                    ear_buckets[2] += 1
                elif v < 0.30:
                    ear_buckets[3] += 1
                else:
                    ear_buckets[4] += 1

            return {
                "labels": labels,
                "alert_counts": alert_counts,
                "blink_counts": blink_counts,
                "danger_count": danger_count + critical_count,
                "warning_count": warning_count,
                "ear_buckets": ear_buckets,
                "ear_labels": ear_labels,
                "total_events": len(events),
                "total_blinks_period": sum(blink_counts)
            }

