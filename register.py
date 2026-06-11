from insightface.app import FaceAnalysis
import cv2
import sqlite3
import pickle
import numpy as np

# ==========================
# CONFIG
# ==========================
# IMAGE_PATH = "images/friend.jpg"
THRESHOLD = 0.6
face_cache = {}
# ==========================
# COSINE SIMILARITY
# ==========================
def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )

# ==========================
# DATABASE SETUP
# ==========================
conn = sqlite3.connect("faces.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS persons(
    person_id INTEGER PRIMARY KEY AUTOINCREMENT,
    embedding BLOB
)
""")

conn.commit()

# ==========================
# LOAD FACE MODEL
# ==========================
app = FaceAnalysis()
app.prepare(ctx_id=-1)

# ==========================
# READ IMAGE
# ==========================
# img = cv2.imread(IMAGE_PATH)
cap = cv2.VideoCapture("videos/test.mp4")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    faces = app.get(frame)
    for face in faces:

        query_embedding = face.embedding

        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox

        face_key = (x1 // 50, y1 // 50)

        if face_key in face_cache:

            label = face_cache[face_key]

        else:

            print("DB SEARCH")

            cursor.execute(
                "SELECT person_id, embedding FROM persons"
            )

            rows = cursor.fetchall()

            best_score = -1
            best_id = None

            for row in rows:

                person_id = row[0]

                stored_embedding = pickle.loads(
                    row[1]
                )

                score = cosine_similarity(
                    query_embedding,
                    stored_embedding
                )

                if score > best_score:

                    best_score = score
                    best_id = person_id

            if best_score > THRESHOLD:

                label = f"ID:{best_id}"

            else:

                label = "UNKNOWN"

            face_cache[face_key] = label

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()