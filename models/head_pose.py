import cv2
import numpy as np

class HeadPoseEstimator:
    """Estimates Euler angles (Yaw, Pitch, Roll) from facial landmarks using PnP."""

    # Canonical 3D facial model points (Nose, Chin, Left Eye, Right Eye, Left Mouth, Right Mouth)
    MODEL_POINTS_3D = np.array([
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),  # Left eye corner
        (225.0, 170.0, -135.0),   # Right eye corner
        (-150.0, -150.0, -125.0), # Left mouth corner
        (150.0, -150.0, -125.0)   # Right mouth corner
    ], dtype=np.float64)

    # Key landmark indices (compatible with MediaPipe Face Mesh)
    LANDMARK_IDS = [1, 152, 263, 33, 291, 61]

    @classmethod
    def estimate_pose(cls, landmarks, image_shape):
        """
        Calculates head orientation angles.
        
        Args:
            landmarks: List of landmark objects with x, y, z normalized coordinates.
            image_shape: Tuple of (height, width)
        Returns:
            dict: {'yaw': float, 'pitch': float, 'roll': float}
        """
        h, w = image_shape[:2]
        image_points = []

        for idx in cls.LANDMARK_IDS:
            lm = landmarks[idx]
            image_points.append((lm.x * w, lm.y * h))

        image_points = np.array(image_points, dtype=np.float64)

        # Camera Intrinsic Matrix Approximation
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Solve Perspective-n-Point
        success, rotation_vec, translation_vec = cv2.solvePnP(
            cls.MODEL_POINTS_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}

        # Convert Rotation Vector to Rotation Matrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)

        # Extract Euler angles from Rotation Matrix
        proj_matrix = np.hstack((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

        pitch = float(euler_angles[0][0])
        yaw = float(euler_angles[1][0])
        roll = float(euler_angles[2][0])

        return {
            'yaw': yaw,
            'pitch': pitch,
            'roll': roll
        }