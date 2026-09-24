import cv2
import numpy as np
from config import Config

class FaceQualityEngine:
    """Evaluates face frames against blur, illumination, and size thresholds."""

    @staticmethod
    def check_blur(image_crop) -> float:
        """Calculates variance of Laplacian to evaluate focus/sharpness."""
        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def check_brightness(image_crop) -> float:
        """Calculates average grayscale intensity."""
        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    @classmethod
    def evaluate_quality(cls, frame, bbox):
        """
        Args:
            frame (np.ndarray): Full BGR image.
            bbox (tuple): (x, y, width, height) of detected face.
        Returns:
            tuple: (is_accepted: bool, score: float, message: str)
        """
        x, y, w, h = bbox
        frame_h, frame_w = frame.shape[:2]

        # 1. Face Size Check
        if w < Config.MIN_FACE_WIDTH_PX:
            return False, 0.0, "Wigira hafi ya kamera (Move closer)"

        # Extract Cropped Region
        x_min, y_min = max(0, x), max(0, y)
        x_max, y_max = min(frame_w, x + w), min(frame_h, y + h)
        face_crop = frame[y_min:y_max, x_min:x_max]

        if face_crop.size == 0:
            return False, 0.0, "Isura ntiyabonetse neza"

        # 2. Sharpness / Blur Filter
        blur_score = cls.check_blur(face_crop)
        if blur_score < Config.MIN_LAPLACIAN_VAR:
            return False, blur_score, "Ifoto iragaragara nka flou, va mu gushyikira (Too blurry)"

        # 3. Brightness / Illumination Filter
        brightness = cls.check_brightness(face_crop)
        if brightness < Config.MIN_BRIGHTNESS:
            return False, brightness, "Urumuri ni rukeya, shyiraho urumuri (Too dark)"
        if brightness > Config.MAX_BRIGHTNESS:
            return False, brightness, "Urumuri ni rwinshi cyane (Overexposed)"

        # Composite Quality Score
        quality_score = min(100.0, (blur_score / Config.MIN_LAPLACIAN_VAR) * 50.0 + (brightness / 128.0) * 50.0)
        return True, round(quality_score, 2), "Quality OK"