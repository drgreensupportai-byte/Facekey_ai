import os
import faiss
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
from config import Config
from database.security import BiometricEncryption
from database.models import UserBiometrics

logger = logging.getLogger("FaceKeyAI.VectorSearch")

class FAISSVectorEngine:
    """
    FAISS IndexFlatIP Search Engine for ultra-fast biometrics match retrieval.
    """

    _index: Optional[faiss.IndexFlatIP] = None
    _user_map: List[int] = []  # Maps index row offset to database user_id

    @classmethod
    def get_index(cls) -> faiss.IndexFlatIP:
        """Returns the active FAISS index instance, initializing or loading if necessary."""
        if cls._index is None:
            cls.load_or_rebuild_index()
        return cls._index

    @classmethod
    def load_or_rebuild_index(cls) -> None:
        """Loads FAISS index from disk or rebuilds it directly from encrypted database templates."""
        index_path = Config.FAISS_INDEX_PATH
        
        if os.path.exists(index_path):
            try:
                cls._index = faiss.read_index(index_path)
                # Load associated ID mapping if available
                map_path = index_path + ".map.npy"
                if os.path.exists(map_path):
                    cls._user_map = np.load(map_path).tolist()
                logger.info(f"Loaded FAISS index with {cls._index.ntotal} vectors from disk.")
                return
            except Exception as e:
                logger.warning(f"Could not load FAISS index from disk ({e}). Rebuilding...")

        cls.rebuild_index_from_db()

    @classmethod
    def rebuild_index_from_db(cls) -> None:
        """
        Query all user encrypted biometric templates from DB, decrypt them, 
        and reconstruct a fresh FAISS IndexFlatIP.
        """
        logger.info("Rebuilding FAISS index from database biometrics...")
        
        # Initialize flat Inner Product index (Cosine Similarity for L2-normalized vectors)
        new_index = faiss.IndexFlatIP(Config.EMBEDDING_DIM)
        new_user_map = []

        try:
            records = UserBiometrics.query.all()
            vectors = []

            for record in records:
                if not record.encrypted_template:
                    continue

                try:
                    # Decrypt template into normalized float32 vector
                    template = BiometricEncryption.decrypt_embedding(record.encrypted_template)
                    
                    if template.ndim == 1:
                        vectors.append(template)
                        new_user_map.append(record.user_id)
                    elif template.ndim == 2: # Handle multi-sample stored templates
                        for vec in template:
                            vectors.append(vec)
                            new_user_map.append(record.user_id)
                except Exception as dec_err:
                    logger.error(f"Failed to decrypt biometric record ID {record.id}: {dec_err}")

            if vectors:
                vector_matrix = np.array(vectors, dtype=np.float32)
                # Ensure all vectors are L2-normalized
                faiss.normalize_L2(vector_matrix)
                new_index.add(vector_matrix)

            cls._index = new_index
            cls._user_map = new_user_map
            cls.save_index()
            logger.info(f"FAISS index rebuild completed. Total vectors indexed: {cls._index.ntotal}")

        except Exception as e:
            logger.error(f"Critical error rebuilding FAISS index: {e}")
            cls._index = faiss.IndexFlatIP(Config.EMBEDDING_DIM)
            cls._user_map = []

    @classmethod
    def save_index(cls) -> None:
        """Persists FAISS index and user_id mapping to disk."""
        if cls._index is not None:
            os.makedirs(os.path.dirname(Config.FAISS_INDEX_PATH), exist_ok=True)
            faiss.write_index(cls._index, Config.FAISS_INDEX_PATH)
            np.save(Config.FAISS_INDEX_PATH + ".map.npy", np.array(cls._user_map))
            logger.info("FAISS index saved to disk.")

    @classmethod
    def add_user_template(cls, user_id: int, template_vector: np.ndarray) -> None:
        """Adds a single user template vector to the index and updates disk persistence."""
        index = cls.get_index()
        
        vec = np.array([template_vector], dtype=np.float32)
        faiss.normalize_L2(vec)

        index.add(vec)
        cls._user_map.append(user_id)
        cls.save_index()

    @classmethod
    def search(cls, query_vector: np.ndarray, top_k: int = None) -> List[Dict[str, Union[int, float]]]:
        """
        Searches the FAISS index for matching candidates given a query embedding.

        Args:
            query_vector (np.ndarray): 1D 512-D float32 normalized vector.
            top_k (int): Number of nearest neighbors to retrieve.
        Returns:
            List[Dict]: Candidates ordered by score [{'user_id': int, 'score': float}]
        """
        index = cls.get_index()
        if index.ntotal == 0:
            return []

        if top_k is None:
            top_k = Config.FAISS_TOP_K

        # Reshape and normalize query vector
        query = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(query)

        # Search index
        scores, indices = index.search(query, min(top_k, index.ntotal))

        results = []
        for sim_score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(cls._user_map):
                results.append({
                    'user_id': cls._user_map[idx],
                    'score': float(sim_score)
                })

        return results