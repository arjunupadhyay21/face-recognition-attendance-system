import cv2
from services.face_recognition import register_employee, recognize_and_mark_attendance
from database.db_operations import get_attendance_by_emp_id, get_attendance_by_name, init_db

# Make sure the database tables exist
init_db()

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam. Please check your camera connection/permissions.")
    exit(1)


def capture_image():
    """
    Opens the webcam feed and waits for the user to press 'c' to capture
    or 'q' to quit without capturing.
    Returns the captured frame, or None if the user quits.
    """
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to read from webcam.")
            return None

        cv2.imshow('Webcam - Press "c" to capture or "q" to quit', frame)

        # Read the key press once so we don't miss events
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            cv2.destroyWindow('Webcam - Press "c" to capture or "q" to quit')
            return frame
        elif key == ord('q'):
            cv2.destroyWindow('Webcam - Press "c" to capture or "q" to quit')
            return None


# Function to fetch attendance by Employee ID
def fetch_attendance(emp_id):
    attendance_records = get_attendance_by_emp_id(emp_id)

    if attendance_records:
        print(f"Attendance records for Employee id {emp_id}:")
        for record in attendance_records:
            print(f"Attendance details: {record}")
    else:
        print(f"No attendance records found for Employee id {emp_id}")


# Main logic to handle options
def main():
    print("1. Register a new employee")
    print("2. Recognize and mark attendance")
    print("3. Check attendance by Employee ID")
    choice = input("Enter your choice (1/2/3): ")

    if choice == "1":
        name = input("Enter employee name: ")
        print("Capturing image for registration...")
        image = capture_image()

        if image is None:
            print("Capture cancelled. Registration aborted.")
        else:
            # Register employee (returns employee_id, or None on failure)
            employee_id = register_employee(name, image)
            if employee_id is not None:
                print(f"Successfully registered {name}. Your Employee ID is: {employee_id}")
            else:
                print("Registration failed.")

    elif choice == "2":
        print("Capturing image for recognition...")
        image = capture_image()

        if image is None:
            print("Capture cancelled. Attendance not marked.")
        else:
            # Recognize and mark attendance (returns (name, employee_id), or None on failure)
            result = recognize_and_mark_attendance(image)
            if result is not None:
                matched_name, matched_id = result
                print(f"Attendance marked for {matched_name} (ID: {matched_id}).")
            else:
                print("Recognition failed or liveness check failed.")

    elif choice == "3":
        emp_id = input("Enter Employee ID to fetch attendance: ")
        fetch_attendance(emp_id)

    else:
        print("Invalid choice.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
