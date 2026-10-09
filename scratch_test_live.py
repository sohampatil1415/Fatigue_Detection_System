import cv2
import time
from app.vision.face_mesh import FaceMeshDetector
from app.vision.drowsiness_detector import DrowsinessDetector

def test():
    fm = FaceMeshDetector()
    detector = DrowsinessDetector("test-session")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("CAMERA 0 CANNOT BE OPENED (may be in use by server)")
        return

    print("Reading 20 frames...")
    for i in range(20):
        ret, frame = cap.read()
        if not ret:
            print(f"Frame {i}: ret False")
            continue
        mesh_res = fm.process_frame(frame)
        ann, metrics = detector.process(frame, fps=15.0)
        print(f"Frame {i}: face={mesh_res['face_detected']}, raw_EAR={mesh_res.get('ear_avg')}, "
              f"smoothed_ear={metrics.get('ear')}, eye_status={metrics.get('eye_status')}, "
              f"state={metrics.get('state')}")
        time.sleep(0.05)
    cap.release()

if __name__ == "__main__":
    test()
