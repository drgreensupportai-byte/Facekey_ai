import json
import logging
import numpy as np
from cryptography.fernet import Fernet
from config import Config
from database.models import db

logger = logging.getLogger("FaceKeyAI.BiometricEncryption")


class BiometricEncryption:
    """
    Handles Fernet (AES-128 in CBC mode with HMAC-SHA256) encryption and decryption 
    for biometric embedding vectors and templates.
    """

    @staticmethod
    def _get_cipher() -> Fernet:
        key = Config.BIOMETRIC_ENCRYPTION_KEY
        if not key:
            raise ValueError(
                "BIOMETRIC_ENCRYPTION_KEY is missing from environment variables. "
                "Please configure a valid Fernet key in your .env file."
            )
        return Fernet(key.encode('utf-8') if isinstance(key, str) else key)

    @classmethod
    def encrypt_embedding(cls, embedding_vector) -> bytes:
        """
        Serializes a NumPy biometric embedding vector or list to JSON and encrypts it into binary format.
        
        Args:
            embedding_vector (np.ndarray | list): Vector of shape (512,) or (N, 512)
        Returns:
            bytes: Encrypted binary biometric template
        """
        cipher = cls._get_cipher()

        # Convert NumPy array or iterable to standard Python nested lists
        if isinstance(embedding_vector, np.ndarray):
            vector_list = embedding_vector.tolist()
        elif isinstance(embedding_vector, list):
            vector_list = embedding_vector
        else:
            vector_list = list(embedding_vector)

        serialized_data = json.dumps(vector_list)
        encrypted_bytes = cipher.encrypt(serialized_data.encode('utf-8'))
        return encrypted_bytes

    @classmethod
    def decrypt_embedding(cls, encrypted_bytes: bytes) -> np.ndarray:
        """
        Decrypts binary template data and deserializes it back into a float32 NumPy array.
        
        Args:
            encrypted_bytes (bytes): Encrypted binary payload
        Returns:
            np.ndarray: Decrypted feature vector(s) formatted as float32
        """
        if not encrypted_bytes:
            raise ValueError("Encrypted biometric payload cannot be empty.")

        cipher = cls._get_cipher()
        decrypted_data = cipher.decrypt(encrypted_bytes).decode('utf-8')
        vector_list = json.loads(decrypted_data)
        
        # Enforce float32 output for FAISS vector indexing compatibility
        vector_array = np.array(vector_list, dtype=np.float32)

        # L2-normalize vectors if 1D array to guarantee accurate Cosine Similarity
        if vector_array.ndim == 1:
            norm = np.linalg.norm(vector_array)
            if norm > 0:
                vector_array = vector_array / norm

        return vector_array


def init_db(app):
    """
    Binds Flask app instance to SQLAlchemy database, creates underlying directory structures, 
    and initializes database tables.
    """
    # Ensure system directories (dataset/, indexes/, database/) exist
    Config.init_app(app)

    db.init_app(app)
    with app.app_context():
        db.create_all()
        logger.info("Database initialized and schema created successfully.")