import sqlite3
import numpy as np

DB_PATH = 'attendance.db'


def init_db():
    """
    Creates the employees and attendance tables if they don't already exist.
    Safe to call every time the app starts.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            face_embedding BLOB NOT NULL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (employee_id) REFERENCES employees(id)
        )
    ''')

    conn.commit()
    conn.close()


def add_employee(name, face_embedding):
    """Adds a new employee with their face embedding. Returns the new employee_id, or None on failure."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        face_embedding_blob = face_embedding.astype(np.float32).tobytes()

        cursor.execute('''
            INSERT INTO employees (name, face_embedding)
            VALUES (?, ?)
        ''', (name, face_embedding_blob))

        employee_id = cursor.lastrowid
        conn.commit()
        return employee_id

    except sqlite3.Error as e:
        print(f"Database error while adding employee: {e}")
        return None
    finally:
        conn.close()


def get_all_employee_embeddings():
    """Returns a list of (id, name, embedding) tuples for all employees. Returns [] on failure."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute('SELECT id, name, face_embedding FROM employees')
        rows = cursor.fetchall()

        employees = [(row[0], row[1], np.frombuffer(row[2], dtype=np.float32)) for row in rows]
        return employees

    except sqlite3.Error as e:
        print(f"Database error while fetching employees: {e}")
        return []
    finally:
        conn.close()


def mark_attendance(employee_id):
    """Records an attendance entry for the given employee_id. Returns True on success, False on failure."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO attendance (employee_id)
            VALUES (?)
        ''', (employee_id,))

        conn.commit()
        return True

    except sqlite3.Error as e:
        print(f"Database error while marking attendance: {e}")
        return False
    finally:
        conn.close()


def get_attendance_by_name(name):
    """Returns attendance timestamps for the given employee name. Returns [] on failure."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT a.timestamp
            FROM attendance a
            JOIN employees e ON a.employee_id = e.id
            WHERE e.name = ?
            ORDER BY a.timestamp DESC
        ''', (name,))

        return cursor.fetchall()

    except sqlite3.Error as e:
        print(f"Database error while fetching attendance by name: {e}")
        return []
    finally:
        conn.close()


def get_attendance_by_emp_id(emp_id):
    """Returns attendance records for the given employee ID. Returns [] on failure."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        query = '''
            SELECT attendance.id, employees.name, attendance.timestamp
            FROM attendance
            JOIN employees ON attendance.employee_id = employees.id
            WHERE employees.id = ?
            ORDER BY attendance.timestamp DESC
        '''

        cursor.execute(query, (emp_id,))
        return cursor.fetchall()

    except sqlite3.Error as e:
        print(f"Database error while fetching attendance by employee ID: {e}")
        return []
    finally:
        conn.close()
