import cv2

from ml_config import MLModelLoader
from models.SignLanguage import CameraOpenError


def main():
    loader = MLModelLoader("ml_config.yml")

    try:
        # 1) Real-time
        loader.model_processor("sign_detect", "run_realtime")
    except CameraOpenError as e:
        print(f"Fatal: {e}")

    # 2) Single-frame
    # frame = cv2.imread("some_test.jpg")
    # letter, conf = loader.model_processor("sign_detect", "predict_frame", frame)
    # print(letter, conf)

if __name__ == "__main__":
    main()
