import cv2
import time
import threading
import logging
import numpy as np

logger = logging.getLogger(__name__)

class CameraManager:
    """
    Thread-safe OpenCV Camera Manager with threaded background reading,
    dynamic resolution switching, FPS calculation, and hardware fallback.
    """

    def __init__(self, camera_index=0, resolution=(640, 480)):
        self.camera_index = camera_index
        self.target_width, self.target_height = resolution
        self.cap = None
        self.is_running = False
        self.is_synthetic = False

        # Threading support
        self.lock = threading.Lock()
        self.read_thread = None
        self.latest_frame = None
        self.ret = False
        self.last_frame_time = time.time()

        # FPS calculation variables
        self.fps = 0.0
        self.frame_count = 0
        self.start_time = time.time()

        self.start_camera()

    def start_camera(self):
        """Initializes OpenCV VideoCapture device or synthetic driver fallback."""
        with self.lock:
            if self.cap is not None and self.cap.isOpened():
                self.is_running = False
                if self.read_thread and self.read_thread.is_alive():
                    self.read_thread.join(timeout=1.0)
                self.cap.release()
                self.cap = None

            self.is_synthetic = False

            # Try CAP_DSHOW on Windows first for fast capture startup
            try:
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(self.camera_index)
            except Exception as e:
                logger.warning(f"VideoCapture initialization exception: {e}")
                self.cap = None

            if self.cap is not None and self.cap.isOpened():
                # Apply resolution
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
                self.cap.set(cv2.CAP_PROP_FPS, 30)

                # Test read
                self.ret, self.latest_frame = self.cap.read()
                if not self.ret or self.latest_frame is None:
                    logger.warning("Camera opened but failed to read initial frame.")

            if self.cap is None or not self.cap.isOpened() or not self.ret:
                logger.warning(f"Hardware camera {self.camera_index} not available. Initializing simulated test feed.")
                self.is_synthetic = True
                self.ret = True
                self.latest_frame = self._generate_synthetic_frame()

            self.is_running = True
            self.start_time = time.time()
            self.frame_count = 0
            self.last_frame_time = time.time()

            # Start background reading thread
            self.read_thread = threading.Thread(target=self._update_frame, daemon=True)
            self.read_thread.start()

            mode = "SYNTHETIC SIMULATOR" if self.is_synthetic else "HARDWARE WEBCAM"
            logger.info(f"Camera manager started ({mode}) on device index {self.camera_index} [{self.target_width}x{self.target_height}]")
            return True

    def _generate_synthetic_frame(self):
        """Generates synthetic test feed if webcam hardware is missing."""
        canvas = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        # Gradient background
        for y in range(self.target_height):
            canvas[y, :, :] = (int(20 + 15 * (y / self.target_height)),
                               int(22 + 10 * (y / self.target_height)),
                               int(32 + 15 * (y / self.target_height)))

        # Center simulated driver face circle
        cx, cy = self.target_width // 2, self.target_height // 2
        cv2.ellipse(canvas, (cx, cy), (120, 160), 0, 0, 360, (190, 210, 240), 2)
        cv2.circle(canvas, (cx - 45, cy - 25), 18, (16, 185, 129), 2)
        cv2.circle(canvas, (cx + 45, cy - 25), 18, (16, 185, 129), 2)
        cv2.circle(canvas, (cx - 45, cy - 25), 6, (34, 211, 238), -1)
        cv2.circle(canvas, (cx + 45, cy - 25), 6, (34, 211, 238), -1)
        cv2.line(canvas, (cx - 30, cy + 60), (cx + 30, cy + 60), (16, 185, 129), 2)

        cv2.putText(canvas, "SIMULATED DRIVER FEED", (cx - 130, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (34, 211, 238), 2)
        cv2.putText(canvas, f"Resolution: {self.target_width}x{self.target_height}", (cx - 80, self.target_height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 175, 195), 1)
        return canvas

    def _update_frame(self):
        """Continuously reads frames into memory buffer."""
        while self.is_running:
            if not self.is_synthetic and self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.ret = ret
                        self.latest_frame = frame
                        self.last_frame_time = time.time()
                else:
                    time.sleep(0.01)
            else:
                # Synthetic simulated feed update
                with self.lock:
                    self.ret = True
                    self.latest_frame = self._generate_synthetic_frame()
                    self.last_frame_time = time.time()
                time.sleep(0.033) # ~30 FPS

    def get_frame(self):
        """Returns the latest frame instantly from buffer and computes FPS."""
        with self.lock:
            if not self.is_running or not self.ret or self.latest_frame is None:
                return False, None

            frame = self.latest_frame.copy()

            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            if elapsed_time >= 1.0:
                self.fps = round(self.frame_count / elapsed_time, 1)
                self.frame_count = 0
                self.start_time = time.time()

            return True, frame

    def set_resolution(self, width, height):
        """Updates capture target resolution."""
        self.target_width = int(width)
        self.target_height = int(height)
        return self.start_camera()

    def switch_camera(self, new_index):
        """Switches video input feed to a new camera index."""
        self.camera_index = int(new_index)
        return self.start_camera()

    def stop_camera(self):
        """Releases video hardware resources cleanly."""
        self.is_running = False
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=1.0)

        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.ret = False
            self.latest_frame = None
            logger.info("Camera manager stopped and resources released.")

    def get_fps(self):
        """Returns current frames per second."""
        return self.fps

    def get_diagnostics(self):
        """Returns camera health status for diagnostics."""
        return {
            "is_running": self.is_running,
            "is_synthetic": self.is_synthetic,
            "camera_index": self.camera_index,
            "resolution": f"{self.target_width}x{self.target_height}",
            "fps": self.fps,
            "seconds_since_last_frame": round(time.time() - self.last_frame_time, 2)
        }
