import os
import uuid
import sqlite3
import pickle
from fastapi import APIRouter, UploadFile, File, Form

from src.config import DB_NAME, FACE_REC_AVAILABLE
from src.database import load_known_faces

if FACE_REC_AVAILABLE:
    import face_recognition

router = APIRouter(prefix="/faces", tags=["Faces"])


@router.post("/register")
async def register_face(
    file: UploadFile = File(...), name: str = Form(...), type: str = Form("blacklist")
):
    if not FACE_REC_AVAILABLE:
        return {"status": "error", "message": "Face Rec not available"}
    temp_filename = f"temp_{uuid.uuid4()}.jpg"
    try:
        with open(temp_filename, "wb") as buffer:
            buffer.write(await file.read())

        image = face_recognition.load_image_file(temp_filename)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            encoding = encodings[0]
            encoding_blob = pickle.dumps(encoding)
            face_id = str(uuid.uuid4())

            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            c.execute(
                "INSERT INTO faces VALUES (?,?,?,?)",
                (face_id, name, type, encoding_blob),
            )
            conn.commit()
            conn.close()

            load_known_faces()  # Reload in-memory profiles
            return {"status": "success", "message": f"Face registered: {name}"}
        else:
            return {"status": "error", "message": "No face found in image"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


@router.get("")
async def get_faces():
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT id, name, type FROM faces")
        rows = c.fetchall()
        conn.close()
        return [{"id": r[0], "name": r[1], "type": r[2]} for r in rows]
    except Exception as e:
        return {"error": str(e)}


@router.delete("/{face_id}")
async def delete_face(face_id: str):
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("DELETE FROM faces WHERE id = ?", (face_id,))
        conn.commit()
        conn.close()
        load_known_faces()  # Reload in-memory profiles
        return {"status": "success", "message": "Face deleted successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
