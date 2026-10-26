from keras_facenet import FaceNet
import numpy as np
import cv2
import dlib

# Load the FaceNet model
embedder = FaceNet()

# Reuse a face detector to crop the face before embedding
# (FaceNet's .embeddings() expects a cropped face, not a full raw frame)
_detector = dlib.get_frontal_face_detector()

REQUIRED_SIZE = (160, 160)  # FaceNet's expected input size


def get_face_embedding(image):
    """
    Detects and crops the face in the given BGR image, then returns its
    FaceNet embedding as a NumPy array. Returns None if no face is found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = _detector(gray)

    if len(faces) == 0:
        print("No face detected in image.")
        return None

    # Use the first detected face
    face = faces[0]
    x1, y1 = max(face.left(), 0), max(face.top(), 0)
    x2, y2 = face.right(), face.bottom()

    face_crop = image[y1:y2, x1:x2]

    if face_crop.size == 0:
        print("Face crop failed (empty region).")
        return None

    # Convert to RGB and resize to FaceNet's expected input size
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, REQUIRED_SIZE)

    try:
        embeddings = embedder.embeddings([face_resized])
        return np.array(embeddings[0])
    except Exception as e:
        print(f"Error generating face embedding: {e}")
        return None
