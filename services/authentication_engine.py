import base64
import cv2
import time
import numpy as np
import logging
from typing import Dict, Any, Optional
from config import Config
from models.face_embedding import FaceEmbeddingExtractor
from services.liveness_verifier import AntiSpoofLivenessVerifier
from vector_search import FAISSVectorEngine
from database.models import User, AuthenticationLog, db

logger = logging.getLogger("FaceKeyAI.AuthenticationEngine")

class SessionDebouncer:
    """In-memory tracker to prevent rapid re-authentication log creation for a single user."""
    _last_auth_cache: Dict[int, float] = {}
    DEBOUNCE_INTERVAL = 10.0  # seconds

    @classmethod
    def is_debounced(cls, user_id: int) -> bool:
        now = time.time()
        last_time = cls._last_auth_cache.get(user_id, 0.0)
        if now - last_time < cls.DEBOUNCE_INTERVAL:
            return True
        cls._last_auth_cache[user_id] = now
        return False

class AuthenticationEngine:
    """
    Executes hands-free multi-modal verification against FAISS index and stores access records.
    """

    @classmethod
    def process_verification_request(cls, image_b64: str, audio_b64: Optional[str] = None) -> Dict[str, Any]:
        """
        Handles end-to-end multi-modal verification.
        
        Steps:
        1. Base64 Image Decode.
        2. Passive Liveness Check.
        3. Feature Embedding Extraction.
        4. FAISS Vector Search & Threshold Match Validation.
        5. Log creation with Session Lock/Debounce.
        """
        # Step 1: Base64 Image Decoding
        image_bgr = cls._decode_base64_image(image_b64)
        if image_bgr is None:
            return {"success": False, "reason": "INVALID_IMAGE", "message": "Failed to decode input image payload."}

        # Step 2: Anti-Spoofing Liveness Evaluation
        is_live, liveness_score, liveness_metrics = AntiSpoofLivenessVerifier.evaluate_liveness(image_bgr)
        if not is_live:
            logger.warning(f"Liveness check failed with score {liveness_score:.3f}")
            return {
                "success": False,
                "reason": "LIVENESS_FAILED",
                "message": "Presentation attack detected or poor quality frame.",
                "liveness_score": liveness_score
            }

        # Step 3: Extract 512-D Normalized Vector
        query_embedding = FaceEmbeddingExtractor.extract_embedding(image_bgr)
        if query_embedding is None:
            return {"success": False, "reason": "NO_FACE_DETECTED", "message": "No face detected in frame."}

        # Step 4: FAISS Vector Engine Match Retrieval
        candidates = FAISSVectorEngine.search(query_embedding, top_k=1)
        if not candidates:
            return {"success": False, "reason": "NO_MATCH", "message": "No biometric matches found in index."}

        top_match = candidates[0]
        matched_user_id = top_match['user_id']
        similarity_score = top_match['score']

        # Enforce strict similarity threshold comparison
        strict_threshold = getattr(Config, 'SIMILARITY_THRESHOLD', 0.65)
        if similarity_score < strict_threshold:
            logger.info(f"Top match user_id={matched_user_id} score={similarity_score:.4f} below threshold ({strict_threshold}).")
            return {
                "success": False,
                "reason": "NO_MATCH",
                "message": "Biometric match score below security threshold.",
                "similarity_score": round(similarity_score, 4)
            }

        # Step 5: User Retrieval & Debounce Handlers
        user = User.query.get(matched_user_id)
        if not user or not user.is_active:
            return {"success": False, "reason": "USER_INACTIVE", "message": "Matched account is inactive."}

        # Check session log debounce
        if not SessionDebouncer.is_debounced(matched_user_id):
            cls._log_auth_attempt(user.id, success=True, score=similarity_score)

        return {
            "success": True,
            "username": user.username,
            "user_id": user.id,
            "similarity_score": round(similarity_score, 4),
            "liveness_score": round(liveness_score, 4),
            "redirect": "/dashboard"
        }

    @staticmethod
    def _decode_base64_image(image_b64: str) -> Optional[np.ndarray]:
        try:
            if "," in image_b64:
                image_b64 = image_b64.split(",")[1]
            image_bytes = base64.b64decode(image_b64)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            logger.error(f"Error decoding base64 image: {e}")
            return None

    @staticmethod
    def _log_auth_attempt(user_id: int, success: bool, score: float):
        try:
            log_entry = AuthenticationLog(
                user_id=user_id,
                success=success,
                similarity_score=score
            )
            db.session.add(log_entry)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            logger.error(f"Failed to record authentication log: {e}")