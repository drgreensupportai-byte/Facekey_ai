import os
import cv2
import numpy as np
import logging
from typing import List, Union, Optional
from config import Config

logger = logging.getLogger("FaceKeyAI.EmbeddingExtractor")

class FaceEmbeddingExtractor:
    """
    Extracts L2-normalized 512-dimensional feature vectors using InsightFace ArcFace models.
    """

    _app_instance = None

    @classmethod
    def _get_insightface_app(cls):
        """Lazy-loads and caches the InsightFace analysis engine."""
        if cls._app_instance is None:
            try:
                import insightface
                from insightface.app import FaceAnalysis

                # Initialize InsightFace with buffalo_l or buffalo_s backbone
                app = FaceAnalysis(
                    name='buffalo_l',
                    root=Config.MODELS_DIR,
                    providers=['CPUExecutionProvider']  # Use ['CUDAExecutionProvider'] if GPU is available
                )
                app.prepare(ctx_id=0, det_size=(640, 640))
                cls._app_instance = app
                logger.info("InsightFace ArcFace model initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to load InsightFace model: {e}")
                raise RuntimeError(f"InsightFace initialization error: {e}")
        return cls._app_instance

    @classmethod
    def extract_embedding(cls, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts and normalizes a 512-D embedding vector from a cropped face image.

        Args:
            face_crop (np.ndarray): BGR image crop containing the face.
        Returns:
            np.ndarray: Normalized 1D array of shape (512,) or None if detection fails.
        """
        if face_crop is None or face_crop.size == 0:
            logger.warning("Empty frame passed to extract_embedding.")
            return None

        app = cls._get_insightface_app()
        
        # Detect and extract embeddings
        faces = app.get(face_crop)
        if not faces:
            logger.warning("No face feature embedding could be extracted from image crop.")
            return None

        # Select the primary (largest) detected face
        primary_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        embedding = primary_face.embedding

        # Verify dimensionality
        if embedding.shape[0] != Config.EMBEDDING_DIM:
            raise ValueError(f"Expected {Config.EMBEDDING_DIM}-D vector, got {embedding.shape[0]}-D.")

        # L2 Normalization
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)

    @classmethod
    def extract_batch_embeddings(cls, image_list: List[np.ndarray]) -> List[np.ndarray]:
        """Extracts normalized embeddings for a list of face image crops."""
        embeddings = []
        for img in image_list:
            emb = cls.extract_embedding(img)
            if emb is not None:
                embeddings.append(emb)
        return embeddings