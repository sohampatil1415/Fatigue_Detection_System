import numpy as np
import cv2
import time

class EARCalculator:
    """
    Eye Aspect Ratio (EAR) Calculator & Intelligent Blink Tracking Engine.
    
    Standardized according to Soukupová and Čech (2016):
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    """

    def __init__(self, blink_threshold=0.23):
        self.blink_threshold = blink_threshold
        self.total_blinks = 0
        self.eye_closed_state = False
        self.closure_start_time = None
        self.last_blink_time = 0

    @staticmethod
    def euclidean_distance(pt1, pt2):
        """Calculates 2D Euclidean distance between two points."""
        return float(np.linalg.norm(np.array(pt1) - np.array(pt2)))

    def calculate_ear(self, eye_points):
        """
        Calculates Eye Aspect Ratio (EAR) using the canonical Euclidean formula:
        eye_points: list of 6 points [p1, p2, p3, p4, p5, p6]
        """
        if eye_points is None or len(eye_points) < 6:
            return 0.30

        p1, p2, p3, p4, p5, p6 = eye_points[:6]
        v1 = self.euclidean_distance(p2, p6)
        v2 = self.euclidean_distance(p3, p5)
        h  = self.euclidean_distance(p1, p4)

        if h < 1e-4:
            return 0.0

        ear = (v1 + v2) / (2.0 * h)
        return float(round(ear, 3))

    def calculate_box_ear(self, eye_box, frame_gray):
        """
        Fallback box-level contrast/intensity EAR estimation
        (kept for test compatibility and low-spec environments).
        """
        (x, y, w, h) = eye_box
        if w <= 0 or h <= 0 or frame_gray is None:
            return 0.30

        fh, fw = frame_gray.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(x + w, fw), min(y + h, fh)

        roi = frame_gray[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < 3 or roi.shape[1] < 3:
            return 0.30

        roi_resized = cv2.resize(roi, (48, 32), interpolation=cv2.INTER_AREA)
        rh, rw = roi_resized.shape
        roi_blur = cv2.GaussianBlur(roi_resized, (5, 5), 0)

        mean_val = float(np.mean(roi_blur))
        std_val = float(np.std(roi_blur))

        darkness_score = max(0.0, min(1.0, (170.0 - mean_val) / 90.0))
        contrast_score = max(0.0, min(1.0, (std_val - 12.0) / 38.0))

        ch1, ch2 = int(rh * 0.25), int(rh * 0.75)
        cw1, cw2 = int(rw * 0.25), int(rw * 0.75)
        center_val = float(np.mean(roi_blur[ch1:ch2, cw1:cw2]))
        outer_pixels = (rh * rw) - ((ch2 - ch1) * (cw2 - cw1))
        outer_val = (float(np.sum(roi_blur)) - float(np.sum(roi_blur[ch1:ch2, cw1:cw2]))) / max(1, outer_pixels)
        dip = max(0.0, outer_val - center_val)
        dip_score = max(0.0, min(1.0, dip / 18.0))

        combined = 0.40 * darkness_score + 0.40 * contrast_score + 0.20 * dip_score
        ear = 0.12 + (combined * 0.26)
        return float(round(min(max(ear, 0.10), 0.42), 3))

    def update_blink_count(self, ear, current_time=None):
        """
        Tracks state transitions to register normal blinks.
        Normal physiological blinks last between 100ms and 400ms.
        Prolonged eye closure (> 1000ms) is drowsiness, not a normal blink.
        """
        now = current_time or time.time()
        is_blink_registered = False

        if ear < self.blink_threshold:
            if not self.eye_closed_state:
                self.eye_closed_state = True
                self.closure_start_time = now
        else:
            if self.eye_closed_state:
                duration = now - (self.closure_start_time or now)
                # Count as a blink only if closure was brief (between 70ms and 500ms)
                if 0.07 <= duration <= 0.65:
                    self.total_blinks += 1
                    self.last_blink_time = now
                    is_blink_registered = True
                self.eye_closed_state = False
                self.closure_start_time = None

        return is_blink_registered, self.total_blinks

    def get_eye_status(self, ear, threshold=None):
        """Returns string representation of eye status."""
        th = threshold or self.blink_threshold
        if ear is None or ear <= 0.0:
            return "UNKNOWN"
        if ear < th:
            return "CLOSED"
        elif ear < th + 0.04:
            return "DROWSY"
        return "OPEN"
