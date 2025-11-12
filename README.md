
# Scalable Face Attendance System

A Flask-based face recognition attendance system. This repo contains tools to create a face dataset, compute embeddings (FAISS index + metadata), train an SVM classifier (optional), and run a Flask app that processes video frames for attendance.

## Repo contents (important)
- `scalable_attendance_system.py` - Flask server + routes.
- `face_dataset_creator.py` - helper to capture faces locally.
- `scalable_face_embeddings.py` - compute embeddings & FAISS index.
- `scalable_train_classifier.py` - optional SVM trainer.
- `templates/` - `home.html`, `report.html`.
- `static/` - images, JS, CSS.
- `models/` - generated (FAISS index, metadata, classifier). **Not stored** in repo by default.

## Quick start (local)
1. Create and activate venv:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux / macOS
   venv\Scripts\activate      # Windows

2. Install dependencies:
pip install -r requirements.txt

3. Build face embeddings + index:
python scalable_face_embeddings.py

4. Train SVM:
python scalable_train_classifier.py

5. Run the server:
python scalable_attendance_system.py

Then open http://127.0.0.1:5000/
