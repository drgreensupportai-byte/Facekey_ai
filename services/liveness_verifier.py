import cv2
import numpy as np
import logging
from typing import Tuple, Dict, Any
from config import Config

logger = logging.getLogger("FaceKeyAI.LivenessVerifier")

class AntiSpoofLivenessVerifier:
    """
    Performs multi-modal texture and optical checks to detect presentation attacks 
    (printed photos, screen replays, and static cutouts).
    """

    @classmethod
    def evaluate_liveness(cls, image_bgr: np.ndarray, landmarks: np.ndarray = None) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Runs passive liveness checks on a captured frame.

        Args:
            image_bgr (np.ndarray): Input frame in BGR format.
            landmarks (np.ndarray): Optional face landmarks array.
        Returns:
            Tuple[bool, float, dict]: (is_live, composite_score, detail_metrics)
        """
        if image_bgr is None or image_bgr.size == 0:
            return False, 0.0, {"error": "Empty frame"}

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Texture Blur & Frequency Inspection (Laplacian Variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(1.0, laplacian_var / 500.0)

        # 2. Reflection & Color Distribution Analysis
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        saturation = hsv[:, :, 1]
        
        # High saturation / low contrast anomalies often correspond to digital screens
        sat_std = float(np.std(saturation))
        val_std = float(np.std(v_channel))
        color_score = min(1.0, (sat_std * val_std) / 1200.0)

        # 3. High-Frequency Pattern Noise (FFT Specular Analysis)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-10)
        
        # Calculate high-frequency energy ratio
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols), np.uint8)
        r = 30
        cv2.circle(mask, (ccol, crow), r, 0, -1)
        high_freq_energy = float(np.mean(magnitude_spectrum[mask == 1]))
        fft_score = max(0.0, min(1.0, 1.0 - (high_freq_energy / 200.0)))

        # Composite score calculation
        composite_score = (0.4 * blur_score) + (0.35 * color_score) + (0.25 * fft_score)
        
        # Apply strict thresholds configured in Config
        liveness_threshold = getattr(Config, 'LIVENESS_THRESHOLD', 0.55)
        is_live = composite_score >= liveness_threshold

        metrics = {
            "laplacian_variance": round(laplacian_var, 2),
            "blur_score": round(blur_score, 3),
            "color_score": round(color_score, 3),
            "fft_score": round(fft_score, 3),
            "composite_score": round(composite_score, 3)
        }

        logger.debug(f"Liveness evaluation: is_live={is_live}, score={composite_score:.3f}, metrics={metrics}")
        return is_live, composite_score, metrics