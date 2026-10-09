import cv2
import numpy as np
from app.vision.face_mesh import FaceMeshDetector

fm = FaceMeshDetector()
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if ret:
    h, w = frame.shape[:2]
    import mediapipe as mp
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detection_result = fm.mp_landmarker.detect(mp_img)
    if detection_result.face_landmarks:
        face_lms = detection_result.face_landmarks[0]
        lms = [(lm.x * w, lm.y * h) for lm in face_lms]

        for name, indices in [("RIGHT", fm.RIGHT_EYE_INDICES), ("LEFT", fm.LEFT_EYE_INDICES)]:
            pts = [lms[i] for i in indices]
            v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
            v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
            horiz = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
            ear = (v1 + v2) / (2.0 * horiz)
            print(f"--- {name} EYE (indices={indices}) ---")
            for i, p in enumerate(pts):
                print(f"  pt{i+1} (idx {indices[i]}): x={p[0]:.1f}, y={p[1]:.1f}")
            print(f"  v1={v1:.2f}, v2={v2:.2f}, horiz={horiz:.2f}, EAR={ear:.4f}")
    else:
        print("No face detected in test frame")
else:
    print("Could not read frame from camera 0")
