import cv2
import numpy as np

class VerificationEngine:
    """Handles anti-spoofing liveness verification and biometric template matching."""

    @staticmethod
    def verify_liveness(image: np.ndarray) -> tuple[bool, str]:
        """
        Performs multi-factor passive liveness detection:
        1. Color Saturation Variance (HSV analysis to detect flat screen feeds or paper prints)
        2. High-Frequency Laplacian Variance (detects blurry images or screen reflection noise)
        """
        if image is None:
            return False, "Isoko y'amashusho itemewe."

        # 1. Convert to HSV color space to inspect skin saturation distribution
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _, s, _ = cv2.split(hsv)

        # Screen displays and paper photos present significantly flatter saturation variances
        sat_std = np.std(s)
        if sat_std < 14.0:
            return False, "Igenzura ry'ubuzima (liveness check) ryanze: Hagaragaye ishusho yo kuri ecran cyangwa ifoto yacapwe."

        # 2. Check frequency-domain texture noise using Laplacian variance
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < 35.0:
            return False, "Kugenzura ubuzima byananiranye: Ishusho iri mu kantu gato. Nyamuneka komeza uyirebe."

        return True, "Kuba ari muzima byemejwe."

    @staticmethod
    def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculates normalized Cosine Similarity. Returns a float score between -1.0 and 1.0."""
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = np.dot(vec1, vec2) / (norm_a * norm_b)
        return float(similarity)