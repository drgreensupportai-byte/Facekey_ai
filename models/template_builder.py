import numpy as np
import logging
from typing import List
from config import Config

logger = logging.getLogger("FaceKeyAI.TemplateBuilder")

class BiometricTemplateBuilder:
    """
    Synthesizes and validates multiple multi-angle biometric vectors into 
    a single representative 512-D template vector for a user.
    """

    @classmethod
    def synthesize_template(cls, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Synthesizes a master embedding vector from multiple multi-angle samples.

        Args:
            embeddings (List[np.ndarray]): List of normalized 512-D vectors captured across angles.
        Returns:
            np.ndarray: Normalized master template vector of shape (512,).
        """
        if not embeddings:
            raise ValueError("Cannot synthesize template from empty embedding list.")

        arr = np.array(embeddings, dtype=np.float32)

        # Enforce batch matrix shape (N, 512)
        if arr.ndim != 2 or arr.shape[1] != Config.EMBEDDING_DIM:
            raise ValueError(f"Invalid embedding dimensions: {arr.shape}. Expected (N, {Config.EMBEDDING_DIM}).")

        # Handle outlier filtering if we have enough sample points
        if arr.shape[0] >= 3:
            # Calculate centroid
            centroid = np.mean(arr, axis=0)
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-10)

            # Compute similarity of each sample to the centroid
            similarities = np.dot(arr, centroid_norm)
            
            # Keep samples within 1.5 standard deviations from median similarity
            median_sim = np.median(similarities)
            std_sim = np.std(similarities)
            valid_mask = similarities >= (median_sim - 1.5 * std_sim)

            filtered_arr = arr[valid_mask]
            if filtered_arr.shape[0] > 0:
                arr = filtered_arr

        # Mean aggregation across valid samples
        mean_vector = np.mean(arr, axis=0)

        # L2 Renormalization to project back onto unit hypersphere
        norm = np.linalg.norm(mean_vector)
        if norm == 0:
            raise ValueError("Synthesized template resulted in zero vector.")

        master_template = mean_vector / norm
        return master_template.astype(np.float32)

    @classmethod
    def compute_similarity(cls, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculates Cosine Similarity / Inner Product between two normalized vectors."""
        vec1_flat = vec1.flatten()
        vec2_flat = vec2.flatten()
        return float(np.dot(vec1_flat, vec2_flat))