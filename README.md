# Face Recognition Based Attendance System

An attendance system that registers people by face and marks their attendance by recognizing them through a webcam. It has a command-line interface and a Gradio web UI (AttendPro), and stores everything locally in SQLite.

## Features

- **Register** a person with a webcam photo; their face embedding is stored in SQLite
- **Mark attendance** by recognizing a face against all registered people
- **Check attendance** records by employee ID
- **Basic liveness check**: a face must be detected with eyes open before registering or marking
- Two interfaces: a terminal menu (`test.py`) and a web UI (`frontend/gradio_app.py`)

## How it works

1. **Face detection:** dlib's frontal face detector finds and crops the face.
2. **Liveness check:** dlib's 68-point landmark model is used to compute the eye aspect ratio (EAR). The face passes if the eyes appear open.
3. **Embedding:** the cropped face is resized to 160x160 and passed through FaceNet (`keras-facenet`) to get an embedding vector.
4. **Matching:** the embedding is compared with stored embeddings using cosine similarity; a match above a threshold marks attendance.
5. **Storage:** employees, embeddings and attendance records are kept in a local SQLite database (`attendance.db`).

## Tech stack

Python 3.10, OpenCV, dlib, TensorFlow / Keras, keras-facenet, SQLite, Gradio

## Project structure

```
.
├── database/
│   ├── db_operations.py   # SQLite reads/writes, init_db()
│   └── db_setup.py        # standalone table creation script
├── frontend/
│   └── gradio_app.py      # web UI
├── models/
│   ├── facenet_model.py   # face cropping + embeddings
│   └── liveness_model.py  # eye-aspect-ratio liveness check
├── services/
│   └── face_recognition.py  # register / recognize logic
├── test.py                # command-line interface
└── requirements.txt
```

## Setup

Tested on macOS (Apple Silicon) with Python 3.10. Windows and Linux are untested; on those, use `tensorflow==2.11.0` instead of `tensorflow-macos` and install dlib for your platform.

**1. Create a Python 3.10 conda environment and install dlib from conda-forge**

Installing dlib with `pip` compiles it from source and can fail with recent CMake versions, so use conda:

```bash
conda create -n facerecog python=3.10 -y
conda activate facerecog
conda install -c conda-forge dlib -y
```

**2. Install the other dependencies**

```bash
pip install -r requirements.txt
```

**3. Download the facial landmark model**

Download `shape_predictor_68_face_landmarks.dat.bz2` from [dlib.net/files](http://dlib.net/files/), unzip it, and place the `.dat` file in a `data/` folder in the project root:

```
data/shape_predictor_68_face_landmarks.dat
```

(The file is about 95 MB, so it is not included in this repository.)

## Usage

Run everything from the project root, with the `facerecog` environment active.

**Web UI**

```bash
python frontend/gradio_app.py
```

Open the local link it prints (usually http://127.0.0.1:7860). Use the tabs to register a person, mark attendance, and check attendance by employee ID.

**Command line**

```bash
python test.py
```

In the webcam window, press `c` to capture or `q` to cancel.

The database file (`attendance.db`) is created automatically on first run. It is listed in `.gitignore` because it contains face embeddings, which are personal data.

## Known limitations

- **Liveness is basic.** It only checks that a face is present with eyes open in a single frame. It does not detect printed photos or screen replays, so it is not a real anti-spoofing defense.
- **Matching picks the first match above the threshold**, not the closest face. The similarity threshold (0.5) has not been tuned on a larger set of people.
- **No access control.** The web UI has no login, and embeddings are stored unencrypted in SQLite.
- Tested with a small number of registered faces only.

## Possible improvements

- Compare against all stored faces and choose the highest similarity
- Tune the threshold using same-person and different-person test pairs
- Multi-frame blink or depth-based liveness detection
- Login for the web UI and encrypted embedding storage
- Delete/update registered people
