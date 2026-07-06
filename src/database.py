import sqlite3
import pickle
import threading
from src.config import DB_NAME, FACE_REC_AVAILABLE

known_face_encodings = []
known_face_names = []
known_face_types = []  # 'blacklist' or 'whitelist'
faces_lock = threading.Lock()


def init_db():
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS alerts
                     (id TEXT PRIMARY KEY, message TEXT, timestamp TEXT, image_path TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS faces
                     (id TEXT PRIMARY KEY, name TEXT, type TEXT, encoding BLOB)""")
        conn.commit()
        conn.close()
        print("Database initialized.")
    except Exception as e:
        print(f"Database error: {e}")


def load_known_faces():
    global known_face_encodings, known_face_names, known_face_types
    if not FACE_REC_AVAILABLE:
        return
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT name, type, encoding FROM faces")
        rows = c.fetchall()

        temp_encodings = []
        temp_names = []
        temp_types = []
        for row in rows:
            name, f_type, encoding_blob = row
            encoding = pickle.loads(encoding_blob)
            temp_encodings.append(encoding)
            temp_names.append(name)
            temp_types.append(f_type)
        conn.close()

        with faces_lock:
            known_face_encodings = temp_encodings
            known_face_names = temp_names
            known_face_types = temp_types

        print(f"Loaded {len(known_face_names)} faces.")
    except Exception as e:
        print(f"Error loading faces: {e}")


# Initialize DB on load
init_db()
load_known_faces()
