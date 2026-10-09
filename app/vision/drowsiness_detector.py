import cv2
import time
import os
import pygame
import numpy as np
import logging
import threading
from collections import deque
from app.config import Config
from app.vision.face_mesh import FaceMeshDetector
from app.vision.ear_calculator import EARCalculator
from app.database import DatabaseManager

logger = logging.getLogger(__name__)

class DetectionState:
    """State Machine Constants."""
    NORMAL = "NORMAL"
    EYES_CLOSED = "EYES_CLOSED"
    HEAD_DOWN = "HEAD_DOWN"
    FACE_MISSING = "FACE_MISSING"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    CAMERA_ERROR = "CAMERA_ERROR"

class DrowsinessDetector:
    """
    Production-Grade AI Driver Drowsiness & Fatigue Detection Engine.
    
    Features:
    - 478 3D MediaPipe Facial Landmarks
    - Continuous Euclidean Eye Aspect Ratio (EAR) with temporal EMA filtering
    - Time-based sustained eye closure analysis with hysteresis debounce
    - 3D Head Pose Pitch estimation relative to calibrated seated baseline
    - Robust Face Absence tracking (distinct from drowsiness)
    - 6-State Formal State Machine: NORMAL, EYES_CLOSED, HEAD_DOWN, FACE_MISSING, LOW_CONFIDENCE, CAMERA_ERROR
    - Priority escalation when multiple fatigue markers coincide
    - Multi-layer audio alarm triggering: Pygame Audio + Windows Hardware Beep + Web Audio
    - Dynamic baseline calibration for personalized driver anatomy
    """

    def __init__(self, session_id="default-session"):
        self.session_id = session_id
        self.face_mesh = FaceMeshDetector()
        self.ear_calc = EARCalculator(blink_threshold=0.23)
        self.db = DatabaseManager()

        # ── Configurable Detection Thresholds (Runtime Modifiable) ──
        self.ear_threshold = 0.23            # Below this = eyes considered closed
        self.eye_closure_duration = 0.9      # Seconds of closure to trigger alert (ignores quick 0.2s blinks)
        self.pitch_drop_threshold = 10.0     # Degrees below neutral pitch to consider head down
        self.head_down_duration = 1.0        # Seconds of head nodding down before alert
        self.face_missing_duration = 2.0     # Seconds without face before out-of-view alert
        self.alarm_enabled = True
        self.alarm_volume = 0.8              # 0.0 to 1.0
        self.alarm_cooldown = 1.0            # Seconds between audio alarms while danger persists

        # ── Debounce Grace Timers (Hysteresis against single-frame flutter) ──
        self.eye_reopen_grace_time = 0.35    # Allow up to 350ms of open-eye jitter before resetting closure timer
        self.eye_reopen_start_time = None
        self.head_up_grace_time = 0.35       # Allow up to 350ms before resetting head-down timer
        self.head_up_start_time = None
        self.face_grace_time = 0.45          # Allow up to 450ms of momentary face loss before wiping timers

        # ── Baseline Calibration ──
        self.baseline_ear = 0.30
        self.baseline_pitch = 0.0
        self.is_calibrated = False
        self.calibration_samples = []
        self.is_calibrating = False
        self._auto_calibration_frames = 0
        self._auto_calibration_buffer = []

        # ── State Machine ──
        self.current_state = DetectionState.NORMAL
        self.alert_priority = "NONE"         # NONE, WARNING, DANGER, CRITICAL
        self.alert_message = "Active & Alert"
        self.is_drowsy = False
        self.drowsy_event_count = 0

        # ── Timers (Elapsed Time Checks, Not Fixed Frame Count) ──
        self.eye_closed_start_time = None
        self.head_down_start_time = None
        self.face_missing_start_time = None
        self.last_face_seen_time = time.time()
        self.last_alarm_time = 0
        self.active_alert_start_time = None

        # ── Head Slump & Hair Detection State ──
        self.last_known_face_box = None
        self.last_face_pitch = 0.0
        self.hair_slump_detected = False
        self.hair_slump_box = None

        # ── EAR Smoothing (EMA) ──
        self._ema_alpha = 0.65               # Higher = faster response to real changes
        self.smoothed_ear = 0.30
        self.ear_history = deque(maxlen=60)
        self.smoothed_pitch = 0.0
        self.pitch_history = deque(maxlen=60)
        self._pitch_ema_alpha = 0.60
        self._debug_frame_count = 0

        # ── Audio Alert Setup ──
        self.alarm_sound = None
        self._init_audio()

        # Database session start
        try:
            self.db.start_session(self.session_id)
        except Exception as e:
            logger.error(f"DB session init error: {e}")

    def _init_audio(self):
        """Initializes Pygame audio mixer for hardware audio alert."""
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            alarm_path = getattr(Config, 'AUDIO_ALARM_PATH', '')
            if os.path.exists(alarm_path):
                self.alarm_sound = pygame.mixer.Sound(alarm_path)
                self.alarm_sound.set_volume(self.alarm_volume)
                logger.info("Audio alert sound loaded successfully.")
            else:
                logger.warning(f"Audio file missing at {alarm_path}")
        except Exception as e:
            logger.warning(f"Pygame audio init warning: {e}")
            self.alarm_sound = None

    def update_settings(self, settings_dict):
        """Updates detection thresholds dynamically from frontend settings."""
        if 'ear_threshold' in settings_dict:
            self.ear_threshold = float(settings_dict['ear_threshold'])
        if 'consec_frames' in settings_dict:
            frames = int(settings_dict['consec_frames'])
            self.eye_closure_duration = max(0.6, min(3.0, frames / 20.0))
        if 'eye_closure_duration' in settings_dict:
            self.eye_closure_duration = float(settings_dict['eye_closure_duration'])
        if 'blink_threshold' in settings_dict:
            self.ear_calc.blink_threshold = float(settings_dict['blink_threshold'])
        if 'pitch_threshold' in settings_dict:
            self.pitch_drop_threshold = float(settings_dict['pitch_threshold'])
        if 'head_down_duration' in settings_dict:
            self.head_down_duration = float(settings_dict['head_down_duration'])
        if 'face_missing_duration' in settings_dict:
            self.face_missing_duration = float(settings_dict['face_missing_duration'])
        if 'alarm_enabled' in settings_dict:
            self.alarm_enabled = bool(settings_dict['alarm_enabled'])
        if 'volume' in settings_dict:
            vol = float(settings_dict['volume']) / 100.0
            self.alarm_volume = max(0.0, min(1.0, vol))
            if self.alarm_sound:
                self.alarm_sound.set_volume(self.alarm_volume)

        logger.info(f"Detector settings updated: EAR_TH={self.ear_threshold}, "
                    f"EyeDur={self.eye_closure_duration}s, PitchDrop={self.pitch_drop_threshold}deg")

    def start_calibration(self):
        """Initiates driver seated position baseline calibration."""
        self.calibration_samples = []
        self.is_calibrating = True
        logger.info("Driver baseline calibration initiated.")

    def trigger_alarm(self):
        """Plays audio alert respecting cooldown and mute settings."""
        if not self.alarm_enabled:
            return
        curr_time = time.time()
        if curr_time - self.last_alarm_time > self.alarm_cooldown:
            self.last_alarm_time = curr_time
            logger.warning(f"[ALARM ACTIVE] state={self.current_state}, "
                           f"ear={self.smoothed_ear:.4f}, pitch={self.smoothed_pitch:.1f}, priority={self.alert_priority}")

            # 1. Hardware Pygame audio playback
            if self.alarm_sound is not None:
                try:
                    self.alarm_sound.play()
                except Exception as e:
                    logger.error(f"Pygame audio playback error: {e}")

            # 2. Windows hardware motherboard/speaker beep fallback in background thread
            def _beep():
                try:
                    import winsound
                    winsound.Beep(1200, 320)
                except Exception:
                    pass
            threading.Thread(target=_beep, daemon=True).start()

    def _detect_hair_head_down(self, frame, curr_time):
        """
        Detects if the driver's head is slumped down with hair / top of head visible in camera
        (when facial landmarks are hidden by downward head slump).
        
        Returns:
            (is_hair_down: bool, hair_box: tuple, confidence: float)
        """
        h, w = frame.shape[:2]

        # 1. Determine Head Search ROI
        if self.last_known_face_box is not None and (curr_time - self.last_face_seen_time < 8.0):
            fx, fy, fw, fh = self.last_known_face_box
            rx = max(0, fx - int(fw * 0.30))
            ry = max(0, fy - int(fh * 0.15))
            rw = min(w - rx, int(fw * 1.60))
            rh = min(h - ry, int(fh * 1.70))
        else:
            # Default driver viewing zone (middle 64% of frame, upper 72%)
            rx = int(w * 0.18)
            ry = int(h * 0.10)
            rw = int(w * 0.64)
            rh = int(h * 0.72)

        roi = frame[ry:ry+rh, rx:rx+rw]
        if roi.size == 0 or rw < 40 or rh < 40:
            return False, None, 0.0

        # 2. Skin tone analysis (Cr: 133-173, Cb: 77-127)
        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
        skin_pixels = cv2.countNonZero(skin_mask)
        skin_ratio = skin_pixels / float(rw * rh)

        # If skin is overwhelmingly visible (face upright facing camera), not top of head
        if skin_ratio > 0.32:
            return False, None, 0.0

        # 3. Hair & Head Mass Analysis
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Dark hair mask (black, dark brown, brown under ambient light)
        dark_hair = (gray < 115).astype(np.uint8) * 255

        # Texture / hair strand mask (Laplacian edges)
        lap = cv2.Laplacian(gray, cv2.CV_8U)
        _, text_mask = cv2.threshold(lap, 18, 255, cv2.THRESH_BINARY)

        # Non-skin candidate
        non_skin = cv2.bitwise_not(skin_mask)
        hair_candidate = cv2.bitwise_and(cv2.bitwise_or(dark_hair, text_mask), non_skin)

        # Morphological operations to bridge strands into a solid head/crown blob
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        hair_blob = cv2.morphologyEx(hair_candidate, cv2.MORPH_CLOSE, k_close)
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        hair_blob = cv2.morphologyEx(hair_blob, cv2.MORPH_OPEN, k_open)

        # 4. Contour Analysis
        contours, _ = cv2.findContours(hair_blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_contour = None
        max_area = 0
        min_head_area = max(1400, int(rw * rh * 0.050))
        max_head_area = int(rw * rh * 0.85)

        for c in contours:
            area = cv2.contourArea(c)
            if min_head_area <= area <= max_head_area:
                bx, by, bw, bh = cv2.boundingRect(c)
                aspect = bw / float(bh) if bh > 0 else 0
                if 0.5 <= aspect <= 2.2 and bw >= 40 and bh >= 40:
                    if area > max_area:
                        max_area = area
                        best_contour = (bx, by, bw, bh)

        if best_contour is not None:
            bx, by, bw, bh = best_contour
            global_box = (rx + bx, ry + by, bw, bh)

            # Confidence scoring
            area_score = min(1.0, max_area / (float(rw * rh) * 0.22))
            transition_bonus = 0.35 if (self.last_known_face_box is not None and curr_time - self.last_face_seen_time < 6.0) else 0.0
            pitch_bonus = 0.25 if abs(getattr(self, 'last_face_pitch', 0.0)) > 5.0 else 0.0

            confidence = min(1.0, max(0.0, area_score * 0.5 + transition_bonus + pitch_bonus))

            if confidence >= 0.40 or max_area > min_head_area * 1.3:
                return True, global_box, round(confidence, 2)

        return False, None, 0.0

    def process(self, frame, fps=30.0):
        """
        Processes one video frame with state machine and time-based heuristics.
        """
        curr_time = time.time()
        h, w = frame.shape[:2]

        if frame is None or frame.size == 0:
            self.current_state = DetectionState.CAMERA_ERROR
            return frame, self._get_metrics(fps)

        # ── 1. Face & Landmark Detection ──
        detection = self.face_mesh.process_frame(frame)
        face_detected = detection.get('face_detected', False)
        confidence = detection.get('confidence', 0.0)

        # ── 2. Face Missing vs Head Slump (Hair Visible) Analysis ──
        if not face_detected:
            # Check if driver's head is slumped down with hair / top of head visible
            is_hair_down, hair_box, hair_conf = self._detect_hair_head_down(frame, curr_time)

            if is_hair_down:
                self.hair_slump_detected = True
                self.hair_slump_box = hair_box
                self.face_missing_start_time = None  # Driver is present, head is slumped!
                self.head_up_start_time = None

                if self.head_down_start_time is None:
                    self.head_down_start_time = curr_time
                head_down_elapsed = curr_time - self.head_down_start_time

                if head_down_elapsed >= self.head_down_duration:
                    self.current_state = DetectionState.HEAD_DOWN
                    self.alert_priority = "CRITICAL"
                    self.alert_message = f"DROWSINESS DETECTED: HEAD SLUMPED DOWN (HAIR VISIBLE, {head_down_elapsed:.1f}s)"
                    self.is_drowsy = True
                    self._handle_drowsy_incident(curr_time, "CRITICAL", head_down_elapsed)
                    self.trigger_alarm()
                else:
                    self.current_state = DetectionState.NORMAL
                    self.alert_priority = "NONE"
                    self.alert_message = f"Head Slumping Down... ({head_down_elapsed:.1f}s / {self.head_down_duration:.1f}s)"
                    self.is_drowsy = False

                annotated = self._render_overlay(frame, detection, fps,
                                                is_closed=False,
                                                is_head_down=True,
                                                head_down_elapsed=head_down_elapsed,
                                                hair_box=hair_box)
                metrics = self._get_metrics(fps)
                metrics['head_pitch'] = -30.0
                metrics['eye_status'] = 'HEAD_DOWN'
                metrics['state'] = self.current_state
                metrics['is_drowsy'] = self.is_drowsy
                metrics['drowsy_reason'] = self.alert_message
                return annotated, metrics

            # Hair slump not detected — regular face missing / out of view handling
            self.hair_slump_detected = False
            self.hair_slump_box = None

            if self.face_missing_start_time is None:
                self.face_missing_start_time = curr_time

            elapsed_missing = curr_time - self.face_missing_start_time

            # Only reset eye/head closure timers if face has been missing longer than grace time
            if elapsed_missing >= self.face_grace_time:
                self.eye_closed_start_time = None
                self.eye_reopen_start_time = None
                self.head_down_start_time = None
                self.head_up_start_time = None

            if elapsed_missing >= self.face_missing_duration:
                self.current_state = DetectionState.FACE_MISSING
                self.alert_priority = "WARNING"
                self.alert_message = f"Face Not Detected / Driver Out of View ({elapsed_missing:.1f}s)"
                self.is_drowsy = False
            else:
                self.current_state = DetectionState.LOW_CONFIDENCE
                self.alert_priority = "NONE"
                self.alert_message = "Tracking Lost — Reacquiring Face..."

            annotated = self._render_overlay(frame, detection, fps)
            return annotated, self._get_metrics(fps)

        # Face detected: reset face missing timer & update last known face position
        self.face_missing_start_time = None
        self.last_face_seen_time = curr_time
        self.last_known_face_box = detection.get('face_box')
        self.last_face_pitch = detection.get('head_pitch', 0.0)
        self.hair_slump_detected = False
        self.hair_slump_box = None

        # Check confidence
        if confidence < 0.35:
            self.current_state = DetectionState.LOW_CONFIDENCE
            self.alert_priority = "NONE"
            self.alert_message = "Low Confidence Tracking"
            annotated = self._render_overlay(frame, detection, fps)
            return annotated, self._get_metrics(fps)

        # ── 3. Extract Biometric Signals ──
        raw_ear = detection.get('ear_avg', 0.30)
        raw_pitch = detection.get('head_pitch', 0.0)

        # Exponential Moving Average (EMA) smoothing — alpha=0.65 gives fast response
        self.smoothed_ear = self._ema_alpha * raw_ear + (1.0 - self._ema_alpha) * self.smoothed_ear
        self.smoothed_ear = round(float(self.smoothed_ear), 4)
        self.ear_history.append(self.smoothed_ear)

        self.smoothed_pitch = self._pitch_ema_alpha * raw_pitch + (1.0 - self._pitch_ema_alpha) * self.smoothed_pitch
        self.smoothed_pitch = round(float(self.smoothed_pitch), 1)
        self.pitch_history.append(self.smoothed_pitch)

        # ── Debug logging every 30 frames ──
        self._debug_frame_count += 1
        if self._debug_frame_count % 30 == 0:
            logger.debug(f"[EAR] raw={raw_ear:.4f} smoothed={self.smoothed_ear:.4f} "
                         f"threshold={self.ear_threshold:.3f} closed={self.smoothed_ear < self.ear_threshold} "
                         f"state={self.current_state}")

        # ── 4. Calibration Handling ──
        if self.is_calibrating:
            self.calibration_samples.append((self.smoothed_ear, self.smoothed_pitch))
            if len(self.calibration_samples) >= 30:
                ears = [s[0] for s in self.calibration_samples]
                pitches = [s[1] for s in self.calibration_samples]
                self.baseline_ear = float(np.median(ears))
                self.baseline_pitch = float(np.median(pitches))
                # Set threshold to 76% of open-eye EAR, clamped between 0.21 and 0.26
                self.ear_threshold = round(max(0.21, min(0.26, self.baseline_ear * 0.76)), 3)
                self.is_calibrated = True
                self.is_calibrating = False
                logger.info(f"Calibration Complete: Baseline EAR={self.baseline_ear:.4f}, "
                            f"Pitch={self.baseline_pitch:.1f}deg, New EAR Threshold={self.ear_threshold}")

        # ── Auto-calibration on startup (collects first 60 stable frames) ──
        if not self.is_calibrated and not self.is_calibrating:
            if 0.22 < raw_ear < 0.60:
                self._auto_calibration_buffer.append(raw_ear)
            if len(self._auto_calibration_buffer) >= 60:
                auto_ear = float(np.median(self._auto_calibration_buffer))
                self.baseline_ear = auto_ear
                new_threshold = round(max(0.21, min(0.26, auto_ear * 0.76)), 3)
                self.ear_threshold = new_threshold
                self.is_calibrated = True
                logger.info(f"Auto-calibration Complete: Baseline EAR={auto_ear:.4f}, "
                            f"New EAR Threshold={new_threshold}")

        # ── 5. Blink Tracking (Non-Blocking) ──
        is_blink, total_blinks = self.ear_calc.update_blink_count(self.smoothed_ear, curr_time)
        if is_blink:
            try:
                self.db.log_blink(self.session_id, self.smoothed_ear)
            except Exception:
                pass

        # ── 6. Independent Condition Evaluations (With Hysteresis Debounce) ──
        # A. Eye Closure Condition
        is_eye_closed = self.smoothed_ear < self.ear_threshold
        if is_eye_closed:
            self.eye_reopen_start_time = None
            if self.eye_closed_start_time is None:
                self.eye_closed_start_time = curr_time
            eye_closure_elapsed = curr_time - self.eye_closed_start_time
        else:
            if self.eye_closed_start_time is not None:
                if self.eye_reopen_start_time is None:
                    self.eye_reopen_start_time = curr_time
                # If eyes reopened for less than grace window (350ms), maintain closure duration
                if (curr_time - self.eye_reopen_start_time) < self.eye_reopen_grace_time:
                    eye_closure_elapsed = curr_time - self.eye_closed_start_time
                else:
                    self.eye_closed_start_time = None
                    self.eye_reopen_start_time = None
                    eye_closure_elapsed = 0.0
            else:
                eye_closure_elapsed = 0.0

        eye_closure_triggered = eye_closure_elapsed >= self.eye_closure_duration

        # B. Head Down Condition (Detects downward head pitch & hair/crown visibility)
        relative_pitch = self.smoothed_pitch - self.baseline_pitch
        # Pitch angle is positive when nodding head down in MediaPipe (+15° to +40°) and negative in solvePnP (-15° to -40°)
        is_pitch_down = (relative_pitch > self.pitch_drop_threshold) or \
                        (relative_pitch < -self.pitch_drop_threshold) or \
                        (self.smoothed_pitch > 14.0) or \
                        (self.smoothed_pitch < -14.0)

        # Hair / Crown visibility check inside tracked face bounding box
        face_box = detection.get('face_box')
        has_hair_prominent = False
        if face_box is not None:
            fx, fy, fw, fh = face_box
            if fw > 30 and fh > 30:
                top_h = int(fh * 0.50)
                upper_roi = frame[max(0, fy):min(h, fy + top_h), max(0, fx):min(w, fx + fw)]
                if upper_roi.size > 0:
                    gray_u = cv2.cvtColor(upper_roi, cv2.COLOR_BGR2GRAY)
                    dark_ratio = np.count_nonzero(gray_u < 115) / float(upper_roi.shape[0] * upper_roi.shape[1])
                    if dark_ratio > 0.38:
                        has_hair_prominent = True

        # Head down triggers if pitch exceeds limit OR dark hair dominates upper face with forward tilt
        is_head_down = is_pitch_down or (has_hair_prominent and (relative_pitch > 6.0 or self.smoothed_pitch > 8.0))

        if is_head_down:
            self.head_up_start_time = None
            if self.head_down_start_time is None:
                self.head_down_start_time = curr_time
            head_down_elapsed = curr_time - self.head_down_start_time
        else:
            if self.head_down_start_time is not None:
                if self.head_up_start_time is None:
                    self.head_up_start_time = curr_time
                if (curr_time - self.head_up_start_time) < self.head_up_grace_time:
                    head_down_elapsed = curr_time - self.head_down_start_time
                else:
                    self.head_down_start_time = None
                    self.head_up_start_time = None
                    head_down_elapsed = 0.0
            else:
                head_down_elapsed = 0.0

        head_down_triggered = head_down_elapsed >= self.head_down_duration

        # ── 7. State Machine Transitions & Priority Escalation ──
        if eye_closure_triggered and head_down_triggered:
            # Multi-condition escalation: Both eyes closed AND head nodding down
            self.current_state = DetectionState.EYES_CLOSED
            self.alert_priority = "CRITICAL"
            self.alert_message = "CRITICAL: EYES CLOSED & HEAD DOWN!"
            self.is_drowsy = True
            self._handle_drowsy_incident(curr_time, "CRITICAL", max(eye_closure_elapsed, head_down_elapsed))
            self.trigger_alarm()

        elif eye_closure_triggered:
            self.current_state = DetectionState.EYES_CLOSED
            self.alert_priority = "DANGER"
            self.alert_message = f"DROWSINESS DETECTED: EYES CLOSED ({eye_closure_elapsed:.1f}s)"
            self.is_drowsy = True
            self._handle_drowsy_incident(curr_time, "DANGER", eye_closure_elapsed)
            self.trigger_alarm()

        elif head_down_triggered:
            self.current_state = DetectionState.HEAD_DOWN
            self.alert_priority = "DANGER"
            if has_hair_prominent or abs(self.smoothed_pitch) >= 14.0:
                self.alert_message = f"DROWSINESS DETECTED: HEAD DOWN (HAIR VISIBLE, {head_down_elapsed:.1f}s)"
            else:
                self.alert_message = f"DROWSINESS DETECTED: HEAD DOWN ({head_down_elapsed:.1f}s)"
            self.is_drowsy = True
            self._handle_drowsy_incident(curr_time, "DANGER", head_down_elapsed)
            self.trigger_alarm()

        elif is_eye_closed and eye_closure_elapsed > 0.25:
            # Pre-alert while closure is building up
            self.current_state = DetectionState.NORMAL
            self.alert_priority = "NONE"
            self.alert_message = f"Eyes Closing... ({eye_closure_elapsed:.1f}s / {self.eye_closure_duration:.1f}s)"
            self.is_drowsy = False
            self.active_alert_start_time = None

        elif is_head_down and head_down_elapsed > 0.25:
            # Pre-alert while head nod is building up
            self.current_state = DetectionState.NORMAL
            self.alert_priority = "NONE"
            self.alert_message = f"Head Nodding Down... ({head_down_elapsed:.1f}s / {self.head_down_duration:.1f}s)"
            self.is_drowsy = False
            self.active_alert_start_time = None

        else:
            self.current_state = DetectionState.NORMAL
            self.alert_priority = "NONE"
            self.alert_message = "Driver Active & Alert"
            self.is_drowsy = False
            self.active_alert_start_time = None

        # ── 8. Render HUD & Return ──
        annotated = self._render_overlay(frame, detection, fps,
                                        is_closed=is_eye_closed,
                                        is_head_down=is_head_down,
                                        raw_ear=raw_ear,
                                        eye_closure_elapsed=eye_closure_elapsed,
                                        head_down_elapsed=head_down_elapsed)
        return annotated, self._get_metrics(fps)

    def _handle_drowsy_incident(self, curr_time, level, duration):
        """Logs incident once per onset to avoid database spamming."""
        if self.active_alert_start_time is None:
            self.active_alert_start_time = curr_time
            self.drowsy_event_count += 1
            try:
                self.db.log_drowsy_event(
                    self.session_id,
                    ear_val=self.smoothed_ear,
                    duration_seconds=round(duration, 2),
                    alert_level=level
                )
            except Exception as e:
                logger.error(f"Error logging drowsiness event: {e}")

    def _render_overlay(self, frame, detection, fps, is_closed=False, is_head_down=False,
                        raw_ear=None, eye_closure_elapsed=0.0, head_down_elapsed=0.0, hair_box=None):
        """Draws advanced futuristic automotive HUD overlay with real-time biometric telemetry."""
        state_info = {
            'is_closed': is_closed,
            'is_head_down': is_head_down,
            'alert_priority': self.alert_priority
        }
        annotated = self.face_mesh.draw_hud(frame, detection, state_info)
        h, w = annotated.shape[:2]

        # Draw Hair Slump Target Reticle (if top of head/hair is detected)
        if hair_box is not None:
            hx, hy, hw_box, hh_box = hair_box
            c_len = min(24, hw_box // 4)
            h_color = (40, 50, 245) if self.alert_priority in ("CRITICAL", "DANGER") else (0, 180, 255)
            # Corner HUD brackets around slumped head
            cv2.line(annotated, (hx, hy), (hx + c_len, hy), h_color, 2)
            cv2.line(annotated, (hx, hy), (hx, hy + c_len), h_color, 2)
            cv2.line(annotated, (hx + hw_box, hy), (hx + hw_box - c_len, hy), h_color, 2)
            cv2.line(annotated, (hx + hw_box, hy), (hx + hw_box, hy + c_len), h_color, 2)
            cv2.line(annotated, (hx, hy + hh_box), (hx + c_len, hy + hh_box), h_color, 2)
            cv2.line(annotated, (hx, hy + hh_box), (hx, hy + hh_box - c_len), h_color, 2)
            cv2.line(annotated, (hx + hw_box, hy + hh_box), (hx + hw_box - c_len, hy + hh_box), h_color, 2)
            cv2.line(annotated, (hx + hw_box, hy + hh_box), (hx + hw_box, hy + hh_box - c_len), h_color, 2)

            # Center target reticle
            cx, cy = hx + hw_box // 2, hy + hh_box // 2
            cv2.circle(annotated, (cx, cy), 14, h_color, 1, lineType=cv2.LINE_AA)
            cv2.circle(annotated, (cx, cy), 3, (0, 240, 255), -1, lineType=cv2.LINE_AA)

            # Floating HUD Banner
            cv2.rectangle(annotated, (hx, hy - 20), (hx + 220, hy - 4), (12, 16, 24), -1)
            cv2.rectangle(annotated, (hx, hy - 20), (hx + 220, hy - 4), h_color, 1)
            cv2.putText(annotated, "HEAD SLUMP: HAIR VISIBLE", (hx + 6, hy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, h_color, 1, cv2.LINE_AA)

        # Top Banner Color based on Priority
        banner_colors = {
            "CRITICAL": (30, 30, 245),   # Red
            "DANGER":   (40, 50, 245),   # Crimson
            "WARNING":  (0, 180, 255),   # Amber/Orange
            "NONE":     (40, 220, 100)   # Cyber Emerald
        }
        b_color = banner_colors.get(self.alert_priority, (40, 220, 100))

        # Flashing Hazard Screen Border when in Danger or Warning
        if self.alert_priority in ("CRITICAL", "DANGER"):
            cv2.rectangle(annotated, (0, 0), (w, h), (40, 50, 245), 5)
            # Pulsing danger text in center-top
            cv2.putText(annotated, "ALARM ACTIVE - WAKE UP!", (w // 2 - 140, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (40, 50, 245), 2, cv2.LINE_AA)
        elif self.alert_priority == "WARNING":
            cv2.rectangle(annotated, (0, 0), (w, h), (0, 180, 255), 4)

        # Top Cockpit Status Bar
        cv2.rectangle(annotated, (0, 0), (w, 46), (12, 16, 24), -1)
        cv2.rectangle(annotated, (0, 42), (w, 46), b_color, -1)
        
        # Status icon circle + text
        cv2.circle(annotated, (22, 23), 6, b_color, -1, lineType=cv2.LINE_AA)
        status_text = f"[{self.current_state}] {self.alert_message.upper()}"
        cv2.putText(annotated, status_text, (36, 29),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, b_color, 2, cv2.LINE_AA)

        # Right corner calibration indicator
        calib_text = "CALIBRATED [LOCKED]" if self.is_calibrated else f"CALIBRATING {len(self._auto_calibration_buffer)}/60"
        calib_color = (60, 230, 100) if self.is_calibrated else (0, 180, 255)
        cv2.putText(annotated, calib_text, (w - 200, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, calib_color, 1, cv2.LINE_AA)

        # Dynamic Eye Closure Progress Meter (when eyes are closing)
        if eye_closure_elapsed > 0.10:
            meter_w, meter_h = 240, 14
            mx, my = (w - meter_w) // 2, 88
            cv2.rectangle(annotated, (mx - 6, my - 18), (mx + meter_w + 6, my + meter_h + 4), (12, 16, 24), -1)
            cv2.rectangle(annotated, (mx - 6, my - 18), (mx + meter_w + 6, my + meter_h + 4), (50, 70, 95), 1)
            cv2.putText(annotated, f"EYE CLOSURE: {eye_closure_elapsed:.1f}s / {self.eye_closure_duration:.1f}s",
                        (mx, my - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 210, 255), 1, cv2.LINE_AA)
            # Fill bar
            pct = min(1.0, eye_closure_elapsed / self.eye_closure_duration)
            fill_len = int(pct * meter_w)
            bar_c = (40, 50, 245) if pct >= 0.85 else ((0, 180, 255) if pct >= 0.45 else (60, 230, 100))
            cv2.rectangle(annotated, (mx, my), (mx + fill_len, my + meter_h), bar_c, -1)
            cv2.rectangle(annotated, (mx, my), (mx + meter_w, my + meter_h), (80, 100, 130), 1)

        # Dynamic Head Nod Progress Meter (when head is nodding down)
        if head_down_elapsed > 0.10 and eye_closure_elapsed <= 0.10:
            meter_w, meter_h = 240, 14
            mx, my = (w - meter_w) // 2, 88
            cv2.rectangle(annotated, (mx - 6, my - 18), (mx + meter_w + 6, my + meter_h + 4), (12, 16, 24), -1)
            cv2.rectangle(annotated, (mx - 6, my - 18), (mx + meter_w + 6, my + meter_h + 4), (50, 70, 95), 1)
            cv2.putText(annotated, f"HEAD NOD: {head_down_elapsed:.1f}s / {self.head_down_duration:.1f}s",
                        (mx, my - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 210, 255), 1, cv2.LINE_AA)
            pct = min(1.0, head_down_elapsed / self.head_down_duration)
            fill_len = int(pct * meter_w)
            bar_c = (40, 50, 245) if pct >= 0.85 else (0, 180, 255)
            cv2.rectangle(annotated, (mx, my), (mx + fill_len, my + meter_h), bar_c, -1)
            cv2.rectangle(annotated, (mx, my), (mx + meter_w, my + meter_h), (80, 100, 130), 1)

        # Bottom-left Cockpit Telemetry HUD Panel
        panel_y = h - 146
        cv2.rectangle(annotated, (12, panel_y), (330, h - 8), (12, 16, 24), -1)
        cv2.rectangle(annotated, (12, panel_y), (330, h - 8), (50, 75, 105), 1)

        eye_status_str = "CLOSED" if is_closed else "OPEN"
        ear_color = (40, 50, 245) if is_closed else (60, 230, 100)

        raw_str = f"{raw_ear:.4f}" if raw_ear is not None else "N/A"
        cv2.putText(annotated, f"EAR  raw:{raw_str}  smth:{self.smoothed_ear:.4f}",
                    (22, panel_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, ear_color, 1, cv2.LINE_AA)
        cv2.putText(annotated, f"Limit:{self.ear_threshold:.3f}   Eyes:{eye_status_str}",
                    (22, panel_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, ear_color, 1, cv2.LINE_AA)

        pitch_color = (40, 50, 245) if is_head_down else (220, 220, 230)
        cv2.putText(annotated, f"Pitch: {self.smoothed_pitch:+.1f}° (Limit: ±{self.pitch_drop_threshold:.0f}°)",
                    (22, panel_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.42, pitch_color, 1, cv2.LINE_AA)

        cv2.putText(annotated, f"Blinks: {self.ear_calc.total_blinks}   Alerts: {self.drowsy_event_count}",
                    (22, panel_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)

        cv2.putText(annotated, f"FPS: {fps:.1f}   Priority: {self.alert_priority}",
                    (22, panel_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 180, 205), 1, cv2.LINE_AA)

        cv2.putText(annotated, f"Alarm Audio: {'ARMED' if self.alarm_enabled else 'MUTED'}",
                    (22, panel_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (60, 230, 100) if self.alarm_enabled else (120, 130, 140), 1, cv2.LINE_AA)

        # Dual Stereo Vertical VU Meters for Eye Aspect Ratio (Right & Left Eyes)
        ear_l = detection.get('ear_left')
        ear_r = detection.get('ear_right')
        ear_l = float(ear_l) if ear_l is not None else float(self.smoothed_ear)
        ear_r = float(ear_r) if ear_r is not None else float(self.smoothed_ear)
        
        # Right eye meter (camera left side)
        bar_h = 120
        rx, lx = 296, 312
        by = panel_y + 10

        for bx, ear_val in [(rx, ear_r), (lx, ear_l)]:
            cv2.rectangle(annotated, (bx, by), (bx + 10, by + bar_h), (25, 30, 40), -1)
            fill = int(min(max(ear_val / 0.42, 0.0), 1.0) * bar_h)
            bar_c = (40, 50, 245) if ear_val < self.ear_threshold else (60, 230, 100)
            cv2.rectangle(annotated, (bx, by + bar_h - fill), (bx + 10, by + bar_h), bar_c, -1)
            cv2.rectangle(annotated, (bx, by), (bx + 10, by + bar_h), (70, 90, 115), 1)
            # Threshold line
            th_y = by + bar_h - int(min(max(self.ear_threshold / 0.42, 0.0), 1.0) * bar_h)
            cv2.line(annotated, (bx - 2, th_y), (bx + 12, th_y), (0, 180, 255), 1)

        cv2.putText(annotated, "R", (rx + 1, by + bar_h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (150, 170, 190), 1)
        cv2.putText(annotated, "L", (lx + 1, by + bar_h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (150, 170, 190), 1)

        return annotated

    def _get_metrics(self, fps):
        """Constructs API metrics dictionary."""
        eye_status = "CLOSED" if self.smoothed_ear < self.ear_threshold else "OPEN"
        if getattr(self, 'hair_slump_detected', False):
            eye_status = "HEAD_DOWN"
        elif not self.face_mesh.face_detected:
            eye_status = "UNKNOWN"

        return {
            "ear": self.smoothed_ear,
            "eye_status": eye_status,
            "is_drowsy": self.is_drowsy,
            "drowsy_reason": self.alert_message,
            "state": self.current_state,
            "alert_priority": self.alert_priority,
            "head_pitch": self.smoothed_pitch,
            "head_pitch_baseline": self.baseline_pitch,
            "ear_threshold": self.ear_threshold,
            "is_calibrated": self.is_calibrated,
            "total_blinks": self.ear_calc.total_blinks,
            "drowsy_events": self.drowsy_event_count,
            "fps": fps
        }
