import numpy as np
from models.facenet_model import get_face_embedding
from models.liveness_model import check_liveness
from database.db_operations import add_employee, get_all_employee_embeddings, mark_attendance

# Cosine similarity threshold for considering two face embeddings a match.
# NOTE: 0.5 is a placeholder. The right value depends on how facenet_model.py
# generates/normalizes embeddings - test with real data (same person across
# multiple photos vs. different people) before relying on this in production.
SIMILARITY_THRESHOLD = 0.5


def register_employee(name, image):
    """
    Registers a new employee using their face image.
    Returns the new employee_id (int) on success, or None on any failure.
    """
    try:
        # Check liveness
        if not check_liveness(image):
            print("Liveness check failed. Registration not allowed.")
            return None

        # Get face embedding
        face_embedding = get_face_embedding(image)

        if face_embedding is None:
            print("Could not detect a face in the image. Registration failed.")
            return None

        # Add employee to the database
        employee_id = add_employee(name, face_embedding)

        if employee_id is None:
            print("Database error: could not register employee.")
            return None

        print(f"Employee {name} registered successfully and Generated Id is: {employee_id}")
        return employee_id

    except Exception as e:
        print(f"Unexpected error during registration: {e}")
        return None


def recognize_and_mark_attendance(image):
    """
    Recognizes a face against registered employees and marks attendance.
    Returns a (name, employee_id) tuple on success, or None on any failure.
    """
    try:
        # Check liveness
        if not check_liveness(image):
            print("Liveness check failed. Access denied.")
            return None

        # Get the embedding of the current face
        current_face_embedding = get_face_embedding(image)

        if current_face_embedding is None:
            print("Could not detect a face in the image. Recognition failed.")
            return None

        # Get all employee embeddings from the database
        employees = get_all_employee_embeddings()

        if not employees:
            print("No registered employees found in the database.")
            return None

        for employee_id, name, stored_embedding in employees:
            similarity = cosine_similarity(current_face_embedding, stored_embedding)
            if similarity > SIMILARITY_THRESHOLD:
                # Mark attendance for the matched employee
                mark_attendance(employee_id)
                print(f"Attendance marked for: Name - {name}, Employee id - {employee_id}")
                return name, employee_id

        print("No match found. Access denied.")
        return None

    except Exception as e:
        print(f"Unexpected error during recognition: {e}")
        return None


def cosine_similarity(embedding1, embedding2):
    """Calculates cosine similarity between two face embeddings."""
    dot_product = embedding1 @ embedding2.T
    norm_embedding1 = np.linalg.norm(embedding1)
    norm_embedding2 = np.linalg.norm(embedding2)

    if norm_embedding1 == 0 or norm_embedding2 == 0:
        return 0.0

    return dot_product / (norm_embedding1 * norm_embedding2)
