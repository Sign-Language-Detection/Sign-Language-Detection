class CameraOpenError(Exception):
    """Raised when the webcam cannot be opened."""
    pass


import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO

class SignLanguageDetector:
    # Class‐level so we only load once
    sign_model_path = None
    yolo = None

    sign_names = [
        'A','B','C','D','E','F','G','H','I','J',
        'K','L','M','N','O','P','Q','R','S','T',
        'U','V','W','X','Y','Z'
    ]

    @classmethod
    def load_model(cls, model_path: str = None):
        if cls.sign_model_path is None:
            base = os.path.dirname(__file__)
            sign_dir = os.path.join(base, "weights")
            cls.sign_model_path = model_path or os.path.join(sign_dir, "sldt.pt")
        if cls.yolo is None:
            cls.yolo = YOLO(cls.sign_model_path)
            cls.yolo.conf = 0.25
            print(f"[INFO] YOLO ASL model loaded from {cls.sign_model_path}")

    @classmethod
    def predict_frame(cls, frame: np.ndarray):
        """
        Run inference on a single BGR frame.
        Returns (label, confidence) or None if no detections.
        """
        cls.load_model()

        # model(frame) returns a list of Results; take the first batch
        results_list = cls.yolo(frame)
        res = results_list[0]  # now a Results object

        # res.boxes is a Boxes object; .conf and .cls are tensors
        confs = res.boxes.conf.cpu().numpy()   # e.g. [0.82, 0.65, ...]
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)  # e.g. [3, 7, ...]

        if confs.size == 0:
            return None

        # pick the highest-confidence detection
        best_idx = int(np.argmax(confs))
        best_conf = float(confs[best_idx])
        label_idx = cls_ids[best_idx]
        label = cls.sign_names[label_idx] if label_idx < len(cls.sign_names) else "Unknown"

        return label, best_conf

    @classmethod
    def run_realtime(cls, camera_idx: int = 0):
        """Open webcam, overlay top-1 sign+conf, quit on 'q'."""
        cls.load_model()
        cap = cv2.VideoCapture(camera_idx)
        if not cap.isOpened():
            raise CameraOpenError(f"Cannot open camera with index {camera_idx}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            res = cls.predict_frame(frame)
            text = f"{res[0]} {res[1]*100:.1f}%" if res else "No sign detected"
            cv2.putText(frame, text, (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("ASL Detector", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
