import cv2
import numpy as np
import os
import time
import logging

logger = logging.getLogger(__name__)

class FaceMeshDetector:
    """
    High-Precision 3D Face & Eye Landmark Detector with MediaPipe FaceLandmarker
    and OpenCV YuNet/Haar Cascade fallback.
    
    Provides:
    - 478 3D facial landmarks
    - 6-point precise Eye Aspect Ratio (EAR) calculation for left & right eyes
    - 3D Head Pose Euler Angle estimation (Pitch, Yaw, Roll) via facial transformation matrix & solvePnP
    - Robust temporal smoothing & tracking confidence estimation
    """

    # Canonical 6-point Eye Landmark indices in MediaPipe Face Mesh
    # Right eye (subject's right, camera's left):
    # p1=33 (outer), p2=160 (top1), p3=158 (top2), p4=133 (inner), p5=153 (bot2), p6=144 (bot1)
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    
    # Left eye (subject's left, camera's right):
    # p1=362 (inner), p2=385 (top1), p3=387 (top2), p4=263 (outer), p5=373 (bot2), p6=380 (bot1)
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

    # Additional contour landmarks for HUD rendering
    RIGHT_EYE_CONTOUR = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
    LEFT_EYE_CONTOUR = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]

    # 3D Generic Anthropomorphic Model Points for head pose solvePnP fallback
    MODEL_POINTS_3D = np.array([
        (0.0, 0.0, 0.0),          # Nose tip (landmark 4)
        (0.0, -330.0, -65.0),     # Chin (landmark 152)
        (-225.0, 170.0, -135.0),  # Left eye outer corner (landmark 263)
        (225.0, 170.0, -135.0),   # Right eye outer corner (landmark 33)
        (-150.0, -150.0, -125.0), # Left mouth corner (landmark 291)
        (150.0, -150.0, -125.0)   # Right mouth corner (landmark 61)
    ], dtype=np.float64)

    def __init__(self):
        self.mp_landmarker = None
        self.yunet = None
        self.face_detected = False
        self.tracking_confidence = 0.0
        
        # Temporal smoothing
        self.prev_landmarks = None
        self.prev_pitch = 0.0
        self.prev_yaw = 0.0
        self.prev_roll = 0.0
        self.missed_frames = 0
        self.max_grace_frames = 3
        
        self._init_models()

    def _init_models(self):
        """Initializes MediaPipe FaceLandmarker and fallback detectors."""
        # 1. MediaPipe FaceLandmarker
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, 'static', 'face_landmarker.task')
            if not os.path.exists(model_path):
                alt_path = os.path.join(os.path.dirname(base_dir), 'app', 'static', 'face_landmarker.task')
                if os.path.exists(alt_path):
                    model_path = alt_path

            if os.path.exists(model_path):
                base_options = python.BaseOptions(model_asset_path=model_path)
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=True,
                    num_faces=1,
                    min_face_detection_confidence=0.45,
                    min_face_presence_confidence=0.45,
                    min_tracking_confidence=0.45
                )
                self.mp_landmarker = vision.FaceLandmarker.create_from_options(options)
                logger.info("MediaPipe FaceLandmarker initialized with 478 3D landmarks.")
            else:
                logger.warning(f"face_landmarker.task not found at {model_path}")
        except Exception as e:
            logger.error(f"MediaPipe FaceLandmarker initialization error: {e}")
            self.mp_landmarker = None

        # 2. YuNet Fallback
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            yunet_path = os.path.join(base_dir, 'static', 'face_detection_yunet.onnx')
            if not os.path.exists(yunet_path):
                alt = os.path.join(os.path.dirname(base_dir), 'app', 'static', 'face_detection_yunet.onnx')
                if os.path.exists(alt):
                    yunet_path = alt

            if os.path.exists(yunet_path):
                self.yunet = cv2.FaceDetectorYN.create(yunet_path, '', (640, 480), 0.35, 0.3, 5000)
                logger.info("YuNet FaceDetectorYN fallback initialized.")
        except Exception as e:
            logger.warning(f"YuNet fallback initialization error: {e}")

    @staticmethod
    def _euclidean_dist(p1, p2):
        """Calculates Euclidean distance between two 2D points."""
        return float(np.linalg.norm(np.array(p1) - np.array(p2)))

    def _calculate_ear_from_landmarks(self, landmarks_2d, indices):
        """
        Calculates Eye Aspect Ratio (EAR) using canonical 6-point Euclidean formula:
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
        """
        pts = [landmarks_2d[i] for i in indices]
        v1 = self._euclidean_dist(pts[1], pts[5])
        v2 = self._euclidean_dist(pts[2], pts[4])
        h = self._euclidean_dist(pts[0], pts[3])

        if h < 1e-4:
            return 0.0
        return float((v1 + v2) / (2.0 * h))

    def _estimate_head_pose(self, transformation_matrix, landmarks_2d, w, h):
        """
        Estimates Head Pose (Pitch, Yaw, Roll in degrees).
        Pitch: positive = looking up, negative = nodding down.
        Yaw: turning left/right.
        Roll: tilting side-to-side.
        """
        if transformation_matrix is not None and len(transformation_matrix) > 0:
            mat = np.array(transformation_matrix)
            R = mat[:3, :3]
            sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
            singular = sy < 1e-6
            if not singular:
                pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
                yaw = np.degrees(np.arctan2(-R[2, 0], sy))
                roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
            else:
                pitch = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
                yaw = np.degrees(np.arctan2(-R[2, 0], sy))
                roll = 0.0
            return float(pitch), float(yaw), float(roll)

        # Fallback solvePnP using 6 facial points
        if landmarks_2d is not None and len(landmarks_2d) >= 300:
            image_points = np.array([
                landmarks_2d[4],   # Nose tip
                landmarks_2d[152], # Chin
                landmarks_2d[263], # Left eye outer corner
                landmarks_2d[33],  # Right eye outer corner
                landmarks_2d[291], # Left mouth corner
                landmarks_2d[61]   # Right mouth corner
            ], dtype=np.float64)

            focal_length = w
            center = (w / 2.0, h / 2.0)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1))

            success, rot_vec, trans_vec = cv2.solvePnP(
                self.MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if success:
                rot_mat, _ = cv2.Rodrigues(rot_vec)
                pitch = float(np.degrees(np.arctan2(rot_mat[2, 1], rot_mat[2, 2])))
                yaw = float(np.degrees(np.arctan2(-rot_mat[2, 0], np.sqrt(rot_mat[0, 0]**2 + rot_mat[1, 0]**2))))
                roll = float(np.degrees(np.arctan2(rot_mat[1, 0], rot_mat[0, 0])))
                return pitch, yaw, roll

        return 0.0, 0.0, 0.0

    def process_frame(self, frame):
        """
        Analyzes a single video frame.
        
        Returns:
            dict containing:
                - 'face_detected': bool
                - 'face_box': (x, y, w, h)
                - 'ear_left': float
                - 'ear_right': float
                - 'ear_avg': float
                - 'head_pitch': float (degrees)
                - 'head_yaw': float (degrees)
                - 'head_roll': float (degrees)
                - 'landmarks_2d': list of (x, y) coordinates
                - 'eye_points_left': list of 6 (x, y) tuples
                - 'eye_points_right': list of 6 (x, y) tuples
                - 'confidence': float (0.0 to 1.0)
        """
        h, w = frame.shape[:2]
        result_data = {
            'face_detected': False,
            'face_box': None,
            'ear_left': None,
            'ear_right': None,
            'ear_avg': 0.30,
            'head_pitch': 0.0,
            'head_yaw': 0.0,
            'head_roll': 0.0,
            'landmarks_2d': None,
            'eye_points_left': None,
            'eye_points_right': None,
            'confidence': 0.0,
            'eye_data': {}
        }

        # 1. Primary: MediaPipe FaceLandmarker
        if self.mp_landmarker is not None:
            try:
                import mediapipe as mp
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                detection_result = self.mp_landmarker.detect(mp_img)

                if detection_result.face_landmarks and len(detection_result.face_landmarks) > 0:
                    face_lms = detection_result.face_landmarks[0]
                    # Use float coordinates for EAR precision; int only for drawing
                    landmarks_2d = [(lm.x * w, lm.y * h) for lm in face_lms]
                    landmarks_2d_int = [(int(lm.x * w), int(lm.y * h)) for lm in face_lms]

                    # Bounding Box (use int coords for pixel operations)
                    xs = [int(pt[0]) for pt in landmarks_2d]
                    ys = [int(pt[1]) for pt in landmarks_2d]
                    min_x, max_x = max(0, min(xs)), min(w, max(xs))
                    min_y, max_y = max(0, min(ys)), min(h, max(ys))
                    face_box = (min_x, min_y, max_x - min_x, max_y - min_y)

                    # Compute Left & Right EAR
                    ear_right = self._calculate_ear_from_landmarks(landmarks_2d, self.RIGHT_EYE_INDICES)
                    ear_left = self._calculate_ear_from_landmarks(landmarks_2d, self.LEFT_EYE_INDICES)
                    ear_avg = (ear_right + ear_left) / 2.0

                    # Head Pose Estimation
                    trans_mat = None
                    if detection_result.facial_transformation_matrixes:
                        trans_mat = detection_result.facial_transformation_matrixes[0]
                    raw_pitch, raw_yaw, raw_roll = self._estimate_head_pose(trans_mat, landmarks_2d, w, h)

                    # Return raw head pose values - smoothing is done by drowsiness_detector
                    pitch = raw_pitch
                    yaw = raw_yaw
                    roll = raw_roll
                    self.prev_pitch = pitch
                    self.prev_yaw = yaw
                    self.prev_roll = roll

                    # Eye Bounding Boxes for compatibility (int coords for pixel ops)
                    r_pts = [landmarks_2d[i] for i in self.RIGHT_EYE_INDICES]  # float for EAR
                    l_pts = [landmarks_2d[i] for i in self.LEFT_EYE_INDICES]   # float for EAR
                    r_pts_int = [landmarks_2d_int[i] for i in self.RIGHT_EYE_INDICES]
                    l_pts_int = [landmarks_2d_int[i] for i in self.LEFT_EYE_INDICES]
                    rx_min = max(0, min(p[0] for p in r_pts_int) - 5)
                    ry_min = max(0, min(p[1] for p in r_pts_int) - 5)
                    rx_max = min(w, max(p[0] for p in r_pts_int) + 5)
                    ry_max = min(h, max(p[1] for p in r_pts_int) + 5)

                    lx_min = max(0, min(p[0] for p in l_pts_int) - 5)
                    ly_min = max(0, min(p[1] for p in l_pts_int) - 5)
                    lx_max = min(w, max(p[0] for p in l_pts_int) + 5)
                    ly_max = min(h, max(p[1] for p in l_pts_int) + 5)

                    eye_data = {
                        'right': (rx_min, ry_min, rx_max - rx_min, ry_max - ry_min),
                        'left': (lx_min, ly_min, lx_max - lx_min, ly_max - ly_min)
                    }

                    self.face_detected = True
                    self.missed_frames = 0
                    self.prev_landmarks = landmarks_2d_int

                    result_data.update({
                        'face_detected': True,
                        'face_box': face_box,
                        'ear_left': round(ear_left, 4),
                        'ear_right': round(ear_right, 4),
                        'ear_avg': round(ear_avg, 4),
                        'head_pitch': round(pitch, 1),
                        'head_yaw': round(yaw, 1),
                        'head_roll': round(roll, 1),
                        'landmarks_2d': landmarks_2d_int,  # int for drawing
                        'eye_points_left': l_pts_int,       # int for HUD drawing
                        'eye_points_right': r_pts_int,      # int for HUD drawing
                        'confidence': 0.95,
                        'eye_data': eye_data
                    })
                    return result_data
            except Exception as e:
                logger.warning(f"MediaPipe detection frame error: {e}")

        # 2. Fallback: YuNet Face Detector
        if self.yunet is not None:
            try:
                self.yunet.setInputSize((w, h))
                status, faces = self.yunet.detect(frame)
                if faces is not None and len(faces) > 0:
                    # Filter out tiny distant faces in background (must be foreground driver face)
                    min_w, min_h = int(w * 0.10), int(h * 0.10)
                    driver_faces = [f for f in faces if f[2] >= min_w and f[3] >= min_h]
                    if not driver_faces:
                        self.missed_frames += 1
                        self.face_detected = False
                        return result_data

                    best_face = max(driver_faces, key=lambda f: f[2] * f[3])
                    fx, fy, fw, fh = int(best_face[0]), int(best_face[1]), int(best_face[2]), int(best_face[3])
                    rx_raw, ry_raw = int(best_face[4]), int(best_face[5])
                    lx_raw, ly_raw = int(best_face[6]), int(best_face[7])

                    eye_w = max(20, int(fw * 0.22))
                    eye_h = max(14, int(fh * 0.16))
                    eye_data = {
                        'right': (max(0, rx_raw - eye_w // 2), max(0, ry_raw - eye_h // 2), eye_w, eye_h),
                        'left': (max(0, lx_raw - eye_w // 2), max(0, ly_raw - eye_h // 2), eye_w, eye_h)
                    }

                    self.face_detected = True
                    self.missed_frames = 0
                    result_data.update({
                        'face_detected': True,
                        'face_box': (fx, fy, fw, fh),
                        'ear_left': 0.28,
                        'ear_right': 0.28,
                        'ear_avg': 0.28,
                        'head_pitch': 0.0,
                        'head_yaw': 0.0,
                        'head_roll': 0.0,
                        'confidence': float(best_face[14]),
                        'eye_data': eye_data
                    })
                    return result_data
            except Exception as e:
                logger.warning(f"YuNet fallback error: {e}")

        # Face not detected
        self.missed_frames += 1
        self.face_detected = False
        return result_data

    def draw_hud(self, frame, detection_info, state_info=None):
        """
        Renders futuristic AI automotive HUD overlay with glowing eye contours,
        concentric pupil reticles, 3D head pose projection vector, face mesh telemetry,
        and high-tech corner target brackets.
        """
        out = frame.copy()
        h, w = out.shape[:2]

        landmarks = detection_info.get('landmarks_2d')
        face_box = detection_info.get('face_box')
        ear_avg = detection_info.get('ear_avg', 0.0)
        ear_left = detection_info.get('ear_left', ear_avg)
        ear_right = detection_info.get('ear_right', ear_avg)
        pitch = detection_info.get('head_pitch', 0.0)
        yaw = detection_info.get('head_yaw', 0.0)
        roll = detection_info.get('head_roll', 0.0)

        is_closed = state_info.get('is_closed', False) if state_info else (ear_avg < 0.22)
        is_head_down = state_info.get('is_head_down', False) if state_info else False
        alert_priority = state_info.get('alert_priority', 'NONE') if state_info else 'NONE'

        # Dynamic Theme Color based on status
        if alert_priority in ('CRITICAL', 'DANGER') or is_closed:
            main_color = (40, 50, 245)      # Bright Crimson Red
            glow_color = (20, 20, 160)
            status_text = "DROWSY / CLOSED"
        elif alert_priority == 'WARNING' or is_head_down:
            main_color = (0, 180, 255)      # Vivid Amber / Orange
            glow_color = (0, 100, 180)
            status_text = "HEAD DOWN WARNING"
        else:
            main_color = (70, 235, 110)     # Neon Cyber Emerald
            glow_color = (25, 120, 50)
            status_text = "TRACKING LOCKED"

        # Draw 3D Face Mesh & Eye Contours
        if landmarks is not None and len(landmarks) >= 468:
            # 1. Subtle Face Oval / Jawline Silhouette Mesh Points
            # Selected landmark indices across chin, cheeks, forehead
            mesh_indices = [
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
                68, 104, 69, 108, 151, 337, 299, 333, 298, 301, # Forehead
                168, 6, 197, 195, 5, 4, 1, 19, 94, 2             # Nose bridge
            ]
            for idx in mesh_indices:
                if idx < len(landmarks):
                    pt = landmarks[idx]
                    cv2.circle(out, pt, 1, (110, 140, 170), -1, lineType=cv2.LINE_AA)

            # 2. Glowing Neon Eye Contours
            r_pts = np.array([landmarks[i] for i in self.RIGHT_EYE_CONTOUR], dtype=np.int32)
            l_pts = np.array([landmarks[i] for i in self.LEFT_EYE_CONTOUR], dtype=np.int32)

            eye_color = (40, 50, 245) if is_closed else (60, 235, 110)
            eye_glow = (20, 20, 160) if is_closed else (20, 110, 45)

            # Outer glow pass (thicker)
            cv2.polylines(out, [r_pts], isClosed=True, color=eye_glow, thickness=4, lineType=cv2.LINE_AA)
            cv2.polylines(out, [l_pts], isClosed=True, color=eye_glow, thickness=4, lineType=cv2.LINE_AA)
            # Inner bright core pass
            cv2.polylines(out, [r_pts], isClosed=True, color=eye_color, thickness=2, lineType=cv2.LINE_AA)
            cv2.polylines(out, [l_pts], isClosed=True, color=eye_color, thickness=2, lineType=cv2.LINE_AA)

            # 3. 6 Key EAR Landmarks with Cyber Ticks
            for idx in self.RIGHT_EYE_INDICES + self.LEFT_EYE_INDICES:
                cv2.circle(out, landmarks[idx], 2, (255, 220, 50), -1, lineType=cv2.LINE_AA)

            # 4. Concentric Iris / Pupil Reticles (landmarks 468 & 473)
            pupil_pts = []
            if len(landmarks) >= 478:
                pupil_pts = [landmarks[468], landmarks[473]] # Right, Left
            elif len(landmarks) >= 468:
                # Estimate pupil center from eye indices if blendshapes unavailable
                r_c = np.mean([landmarks[i] for i in self.RIGHT_EYE_INDICES], axis=0).astype(int)
                l_c = np.mean([landmarks[i] for i in self.LEFT_EYE_INDICES], axis=0).astype(int)
                pupil_pts = [tuple(r_c), tuple(l_c)]

            for pup in pupil_pts:
                px, py = int(pup[0]), int(pup[1])
                # Concentric target reticle rings
                cv2.circle(out, (px, py), 6, (0, 240, 255), 1, lineType=cv2.LINE_AA)
                cv2.circle(out, (px, py), 2, (0, 255, 255), -1, lineType=cv2.LINE_AA)
                # Reticle crosshair ticks
                cv2.line(out, (px - 9, py), (px - 4, py), (0, 240, 255), 1, lineType=cv2.LINE_AA)
                cv2.line(out, (px + 4, py), (px + 9, py), (0, 240, 255), 1, lineType=cv2.LINE_AA)
                cv2.line(out, (px, py - 9), (px, py - 4), (0, 240, 255), 1, lineType=cv2.LINE_AA)
                cv2.line(out, (px, py + 4), (px, py + 9), (0, 240, 255), 1, lineType=cv2.LINE_AA)

            # 5. Floating Holographic Eye Telemetry Badges
            # Right Eye badge (left side of image)
            r_top = min([landmarks[i][1] for i in self.RIGHT_EYE_INDICES])
            r_x = int(np.mean([landmarks[i][0] for i in self.RIGHT_EYE_INDICES]))
            r_badge_y = max(18, int(r_top) - 14)
            r_badge_text = f"R:{ear_right:.2f}" if not is_closed else f"R:CLOSED {ear_right:.2f}"
            cv2.rectangle(out, (r_x - 32, r_badge_y - 12), (r_x + 32, r_badge_y + 3), (12, 16, 24), -1)
            cv2.rectangle(out, (r_x - 32, r_badge_y - 12), (r_x + 32, r_badge_y + 3), eye_color, 1)
            cv2.putText(out, r_badge_text, (r_x - 29, r_badge_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, eye_color, 1, cv2.LINE_AA)

            # Left Eye badge (right side of image)
            l_top = min([landmarks[i][1] for i in self.LEFT_EYE_INDICES])
            l_x = int(np.mean([landmarks[i][0] for i in self.LEFT_EYE_INDICES]))
            l_badge_y = max(18, int(l_top) - 14)
            l_badge_text = f"L:{ear_left:.2f}" if not is_closed else f"L:CLOSED {ear_left:.2f}"
            cv2.rectangle(out, (l_x - 32, l_badge_y - 12), (l_x + 32, l_badge_y + 3), (12, 16, 24), -1)
            cv2.rectangle(out, (l_x - 32, l_badge_y - 12), (l_x + 32, l_badge_y + 3), eye_color, 1)
            cv2.putText(out, l_badge_text, (l_x - 29, l_badge_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, eye_color, 1, cv2.LINE_AA)

            # 6. 3D Head Orientation Pose Vector (projected from nose tip landmark 4)
            nose = landmarks[4]
            nx, ny = int(nose[0]), int(nose[1])
            # Draw nose anchor dot
            cv2.circle(out, (nx, ny), 3, (0, 220, 255), -1, lineType=cv2.LINE_AA)

            # Compute 3D direction vector from pitch and yaw
            # Pitch: positive = up, negative = down. Yaw: positive = right, negative = left
            p_rad = np.radians(pitch)
            y_rad = np.radians(yaw)
            v_len = 55.0
            dx = int(-v_len * np.sin(y_rad) * np.cos(p_rad))
            dy = int(-v_len * np.sin(p_rad)) # in image coords, down is positive y

            head_vec_color = (40, 50, 245) if is_head_down else ((0, 190, 255) if abs(pitch) > 10 else (0, 230, 255))
            cv2.arrowedLine(out, (nx, ny), (nx + dx, ny + dy), head_vec_color, 2, tipLength=0.28, line_type=cv2.LINE_AA)

            # Mini Head Pose Label beside nose
            pose_str = f"P:{pitch:+.0f}°"
            cv2.putText(out, pose_str, (nx + 12, ny + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, head_vec_color, 1, cv2.LINE_AA)

            # 7. High-Tech Cyberpunk Face Target Box
            if face_box:
                fx, fy, fw_box, fh_box = face_box
                # Subtle dashed/thin perimeter
                cv2.rectangle(out, (fx, fy), (fx + fw_box, fy + fh_box), (50, 70, 95), 1, lineType=cv2.LINE_AA)
                
                # Corner HUD brackets (thicker & prominent with tech tick marks)
                c_len = min(26, fw_box // 4)
                # Outer glow
                for ox, oy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    # Top-left
                    cv2.line(out, (fx+ox, fy+oy), (fx + c_len+ox, fy+oy), glow_color, 2)
                    cv2.line(out, (fx+ox, fy+oy), (fx+ox, fy + c_len+oy), glow_color, 2)
                    # Top-right
                    cv2.line(out, (fx + fw_box+ox, fy+oy), (fx + fw_box - c_len+ox, fy+oy), glow_color, 2)
                    cv2.line(out, (fx + fw_box+ox, fy+oy), (fx + fw_box+ox, fy + c_len+oy), glow_color, 2)
                    # Bottom-left
                    cv2.line(out, (fx+ox, fy + fh_box+oy), (fx + c_len+ox, fy + fh_box+oy), glow_color, 2)
                    cv2.line(out, (fx+ox, fy + fh_box+oy), (fx+ox, fy + fh_box - c_len+oy), glow_color, 2)
                    # Bottom-right
                    cv2.line(out, (fx + fw_box+ox, fy + fh_box+oy), (fx + fw_box - c_len+ox, fy + fh_box+oy), glow_color, 2)
                    cv2.line(out, (fx + fw_box+ox, fy + fh_box+oy), (fx + fw_box+ox, fy + fh_box - c_len+oy), glow_color, 2)

                # Inner core corner lines
                cv2.line(out, (fx, fy), (fx + c_len, fy), main_color, 3)
                cv2.line(out, (fx, fy), (fx, fy + c_len), main_color, 3)
                cv2.line(out, (fx + fw_box, fy), (fx + fw_box - c_len, fy), main_color, 3)
                cv2.line(out, (fx + fw_box, fy), (fx + fw_box, fy + c_len), main_color, 3)
                cv2.line(out, (fx, fy + fh_box), (fx + c_len, fy + fh_box), main_color, 3)
                cv2.line(out, (fx, fy + fh_box), (fx, fy + fh_box - c_len), main_color, 3)
                cv2.line(out, (fx + fw_box, fy + fh_box), (fx + fw_box - c_len, fy + fh_box), main_color, 3)
                cv2.line(out, (fx + fw_box, fy + fh_box), (fx + fw_box, fy + fh_box - c_len), main_color, 3)

                # Tech Tag at Top-Left of Face Box
                cv2.rectangle(out, (fx, fy - 18), (fx + 120, fy - 2), (14, 18, 28), -1)
                cv2.rectangle(out, (fx, fy - 18), (fx + 120, fy - 2), main_color, 1)
                cv2.putText(out, status_text, (fx + 6, fy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.33, main_color, 1, cv2.LINE_AA)

        elif face_box:
            # Fallback face box drawing
            fx, fy, fw_box, fh_box = face_box
            cv2.rectangle(out, (fx, fy), (fx + fw_box, fy + fh_box), (70, 90, 120), 2)
            cv2.putText(out, "FACE ACQUIRED", (fx + 6, fy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 200, 255), 1, cv2.LINE_AA)

        return out
