import os
import cv2
import numpy as np

class FaceQualityEngine:
    """Ikora ibijyanye no gutahura mu maso, kugenzura ireme, no gukuramo ibiranga by’ingenzi mu buryo buhamye."""

    @staticmethod
    def _get_cascade():
        cascade_filename = 'haarcascade_frontalface_default.xml'
        if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
            path = os.path.join(cv2.data.haarcascades, cascade_filename)
            if os.path.exists(path):
                return cv2.CascadeClassifier(path)
        return cv2.CascadeClassifier(cv2.samples.findFile(f'haarcascades/{cascade_filename}'))

    @staticmethod
    def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
        nparr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    @classmethod
    def assess_quality(cls, image: np.ndarray):
        if image is None:
            return False, "Amakuru y'ishusho atemewe."

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        mean_brightness = np.mean(gray)
        if mean_brightness < 30:
            return False, "Ahantu hari umwijima cyane. Nyamuneka ongera urumuri."
        if mean_brightness > 230:
            return False, "Ahantu hari urumuri rwinshi cyane."

        face_cascade = cls._get_cascade()
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        if len(faces) == 0:
            return False, "Nta sura yagaragaye mu ifoto ya kamera."
        if len(faces) > 1:
            return False, "Habonetse amasura menshi. Reba neza ko hari umuntu umwe gusa mu ifoto."

        return True, "Byujuje ibisabwa ku bijyanye n'ubuziranenge."

    @classmethod
    def extract_embedding(cls, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cls._get_cascade()
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        if len(faces) == 0:
            raise ValueError("Nta sura yabonetse mu gihe cyo gukuramo.")

        (x, y, w, h) = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        resized_face = cv2.resize(face_roi, (128, 128))

        # Histogram equalization to mitigate shadows and ambient lighting differences
        equalized_face = cv2.equalizeHist(resized_face)

        # HOG descriptor extraction
        win_size = (128, 128)
        block_size = (32, 32)
        block_stride = (16, 16)
        cell_size = (16, 16)
        nbins = 9
        
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        descriptor = hog.compute(equalized_face).flatten().astype(np.float32)

        # Dynamic interpolation to 128-dimensional vector
        target_dim = 128
        orig_indices = np.linspace(0, 1, len(descriptor))
        target_indices = np.linspace(0, 1, target_dim)
        embedding = np.interp(target_indices, orig_indices, descriptor)

        # L2 Normalization
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding