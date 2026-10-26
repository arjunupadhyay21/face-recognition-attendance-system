import dlib
import cv2
import numpy as np
import os

# Load the pre-trained shape predictor model
shape_predictor_path = 'data/shape_predictor_68_face_landmarks.dat'

if not os.path.exists(shape_predictor_path):
    raise FileNotFoundError(
        f"Shape predictor file not found at '{shape_predictor_path}'. "
        "Download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 "
        "and place it in the 'data' folder."
    )

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)

# Eyes-open threshold. EAR is typically ~0.25-0.3 when open, drops below ~0.2 when closed.
# This is a basic anti-photo check (rejects clearly closed-eye / non-face images),
# NOT full blink-sequence liveness detection.
EAR_OPEN_THRESHOLD = 0.22


def check_liveness(image):
    """
    Checks that a face is present and the eyes appear open in the given frame.
    Returns True if a live-looking face is detected, False otherwise.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return False  # No face detected

    for face in faces:
        shape = predictor(gray, face)
        left_eye_ratio = calculate_eye_aspect_ratio(shape, [36, 37, 38, 39, 40, 41])
        right_eye_ratio = calculate_eye_aspect_ratio(shape, [42, 43, 44, 45, 46, 47])
        avg_ratio = (left_eye_ratio + right_eye_ratio) / 2.0

        if avg_ratio > EAR_OPEN_THRESHOLD:
            return True  # Eyes open - treat as live

    return False  # Eyes appear closed, or no valid landmarks


def calculate_eye_aspect_ratio(landmarks, eye_points):
    """
    Calculates the Eye Aspect Ratio (EAR) using full 2D (x, y) landmark positions.
    eye_points should be 6 dlib landmark indices: [left_corner, top1, top2, right_corner, bottom1, bottom2]
    """
    def point(idx):
        p = landmarks.part(idx)
        return np.array([p.x, p.y])

    p1, p2, p3, p4, p5, p6 = [point(i) for i in eye_points]

    # Vertical distances
    A = np.linalg.norm(p2 - p6)
    B = np.linalg.norm(p3 - p5)
    # Horizontal distance
    C = np.linalg.norm(p1 - p4)

    if C == 0:
        return 0.0

    ear = (A + B) / (2.0 * C)
    return ear
