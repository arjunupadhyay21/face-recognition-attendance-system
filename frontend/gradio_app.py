import sys
import os
import cv2
import numpy as np
import gradio as gr
import sqlite3

# Ensure the project root directory is added to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from services.face_recognition import register_employee, recognize_and_mark_attendance
from database.db_operations import init_db

# Make sure the database tables exist (attendance.db is not stored in the repo)
init_db()


def to_bgr(image):
    """Gradio gives RGB images, but the face pipeline expects OpenCV's BGR order."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


# Function to handle submission of registration
def submit_registration(image, name):
    if image is None:
        return "No image to submit."
    if name is None or len(name.strip()) == 0:
        return "No name is given. Please enter the valid employee name."

    employee_id = register_employee(name.strip(), to_bgr(image))
    if employee_id is not None:
        return f"Successfully registered {name}. Employee ID: {employee_id}"
    else:
        return "Registration failed (no face found, eyes closed, or liveness check failed). Try again."


# Function to handle attendance marking
def mark_attendance(image):
    if image is None:
        return "No image captured. Please try again."

    result = recognize_and_mark_attendance(to_bgr(image))
    if result is None:
        return "Face not recognized, or liveness check failed. Please try again."

    name, employee_id = result
    return f"Attendance marked for {name} (ID: {employee_id})"

# Function to get attendance by employee ID
def get_attendance_by_emp_id(emp_id):
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()

    query = '''
    SELECT attendance.id, employees.name, attendance.timestamp
    FROM attendance
    JOIN employees ON attendance.employee_id = employees.id
    WHERE employees.id = ?
    ORDER BY attendance.timestamp DESC;
    '''

    cursor.execute(query, (emp_id,))
    records = cursor.fetchall()
    conn.close()

    if records:
        attendance_str = f"Attendance for Employee ID: {emp_id}\n\n"
        for record in records:
            attendance_str += f"Attendance ID: {record[0]}, Name: {record[1]}, Timestamp: {record[2]}\n"
        return attendance_str
    else:
        return f"No attendance records found for Employee ID: {emp_id}."

# Create the Gradio Interface
def create_interface():
    with gr.Blocks(css=".gr-tab-title { font-weight: bold; }") as interface:
        # Title with styling
        gr.HTML("<h1 style='text-align: center; font-size: 2.5em; font-weight: bold; color: #4CAF50;'>AttendPro</h1>")

        # Section for registering an employee
        with gr.Tab("Register Employee"):
            gr.Markdown("## Register a New Employee")

            # Input for employee name
            name = gr.Textbox(label="Employee Name", placeholder="Enter employee's name")

            # Webcam-related elements
            webcam_input = gr.Image(label="Webcam", visible=False)

            # Cancel and Submit buttons
            cancel_btn = gr.Button("Cancel", visible=False)
            submit_btn = gr.Button("Submit", visible=False)

            # Button to trigger webcam
            register_face_btn = gr.Button("Register Face")
            output_textbox = gr.Textbox(label="Output", visible=False)

            # Event to trigger webcam feed
            register_face_btn.click(
                lambda: [gr.update(value=None, visible=True), gr.update(visible=True), gr.update(visible=True)],
                inputs=None,
                outputs=[webcam_input, webcam_input, cancel_btn]
            )

            # Event when a photo is captured
            webcam_input.change(
                lambda image: [gr.update(visible=True), gr.update(visible=False)],
                inputs=webcam_input,
                outputs=[submit_btn, register_face_btn]
            )

            # Submit button event
            submit_btn.click(
                submit_registration,
                inputs=[webcam_input, name],
                outputs=output_textbox
            )

            submit_btn.click(
                lambda: gr.update(visible=True),
                inputs=None,
                outputs=[output_textbox]
            )

            # Cancel button event
            cancel_btn.click(
                lambda: [gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),
                         gr.update(visible=False), gr.update(visible=False)],
                inputs=None,
                outputs=[webcam_input, cancel_btn, register_face_btn, submit_btn, output_textbox]
            )

        # Section for marking attendance
        with gr.Tab("Mark Attendance"):
            gr.Markdown("## Mark Attendance")

            # Webcam for attendance
            attendance_webcam = gr.Image(label="Webcam", visible=False)

            # Buttons
            mark_attendance_btn = gr.Button("Mark Attendance", elem_classes="btn-primary")
            cancel_attendance_btn = gr.Button("Cancel", visible=False, elem_classes="btn-primary")
            submit_attendance_btn = gr.Button("Submit", visible=False, elem_classes="btn-primary")
            attendance_output_textbox = gr.Textbox(label="Output", visible=False)

            # Event to trigger webcam feed
            mark_attendance_btn.click(
                lambda: [gr.update(value=None, visible=True), gr.update(visible=True), gr.update(visible=True)],
                inputs=None,
                outputs=[attendance_webcam, attendance_webcam, cancel_attendance_btn]
            )

            # Event when a photo is captured
            attendance_webcam.change(
                lambda image: [gr.update(visible=True), gr.update(visible=False)],
                inputs=attendance_webcam,
                outputs=[submit_attendance_btn, mark_attendance_btn]
            )

            # Submit button event
            submit_attendance_btn.click(
                mark_attendance,
                inputs=[attendance_webcam],
                outputs=attendance_output_textbox
            )

            submit_attendance_btn.click(
                lambda: gr.update(visible=True),
                inputs=None,
                outputs=[attendance_output_textbox]
            )

            # Cancel button event
            cancel_attendance_btn.click(
                lambda: [gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),
                         gr.update(visible=False), gr.update(visible=False)],
                inputs=None,
                outputs=[attendance_webcam, cancel_attendance_btn, mark_attendance_btn, submit_attendance_btn,
                         attendance_output_textbox]
            )

        # Section for checking attendance
        with gr.Tab("Check Attendance"):
            gr.Markdown("## Check Attendance")

            # Input for employee ID
            emp_id_input = gr.Textbox(label="Employee ID", placeholder="Enter employee ID")

            # Button to fetch attendance
            check_attendance_btn = gr.Button("Check Attendance", elem_classes="btn-primary")

            # Output for attendance details
            attendance_details_output = gr.Textbox(label="Attendance Details", visible=False)

            # Event to fetch and display attendance details
            check_attendance_btn.click(
                get_attendance_by_emp_id,
                inputs=emp_id_input,
                outputs=attendance_details_output
            )

            # Show the attendance details output after the button click
            check_attendance_btn.click(
                lambda: gr.update(visible=True),
                inputs=None,
                outputs=[attendance_details_output]
            )

    return interface

# Create and launch the interface (run from the project root: python frontend/gradio_app.py)
if __name__ == "__main__":
    interface = create_interface()
    interface.launch()
