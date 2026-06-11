from insightface.app import FaceAnalysis
from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2
import sqlite3
import pickle
best_face_quality = {}
best_face_embedding = {}
best_face_crop = {}
def cosine_similarity(a, b):

    return np.dot(a, b) / (
        np.linalg.norm(a)
        *
        np.linalg.norm(b)
    )
# ---------------------
# Models
# ---------------------
face_app = FaceAnalysis()
face_app.prepare(ctx_id=-1)

yolo = YOLO("yolov8n.pt")

tracker = sv.ByteTrack()
# ---------------------
# Database
# ---------------------
conn = sqlite3.connect(
    "faces.db"
)

cursor = conn.cursor()

THRESHOLD = 0.6
# ---------------------
# Tracking Data
# ---------------------
track_frame_count = {}
track_to_person = {}
active_tracks = set()
# ---------------------
# Video
# ---------------------
cap = cv2.VideoCapture("videos/test.mp4")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # ---------------------
    # Person Detection
    # ---------------------
    results = yolo(frame,verbose=False)[0]

    detections = sv.Detections.from_ultralytics(
        results
    )

    detections = tracker.update_with_detections(
        detections
    )

    # ---------------------
    # Face Detection
    # ---------------------
    faces = face_app.get(frame)

    # ---------------------
    # Track Processing
    # ---------------------
    current_tracks = set()
    for bbox, track_id in zip(
        detections.xyxy,
        detections.tracker_id
    ):

        if track_id is None:
            continue
        current_tracks.add(track_id)
        # New Track
        if track_id not in track_frame_count:

            track_frame_count[track_id] = 1

            print(
                f"New Track {track_id}"
            )

        else:

            track_frame_count[track_id] += 1

        x1, y1, x2, y2 = map(
            int,
            bbox
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            f"Track {track_id} | Frames:{track_frame_count[track_id]}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    # ---------------------
    # Draw Face Boxes
    # ---------------------
    for face in faces:

        fx1, fy1, fx2, fy2 = face.bbox.astype(int)
        face_width = fx2 - fx1
        face_height = fy2 - fy1

        face_area = face_width * face_height
        # Ignore tiny faces
        if face_area < 1500:
            continue
        # ---------------------
        # Face Quality
        # ---------------------
        left_eye = face.kps[0]
        right_eye = face.kps[1]
        nose = face.kps[2]
        left_mouth = face.kps[3]
        right_mouth = face.kps[4]
        # ---------------------
        # Distances
        # ---------------------
        left_eye_nose = np.linalg.norm(
            left_eye - nose
        )

        right_eye_nose = np.linalg.norm(
            right_eye - nose
        )

        left_mouth_nose = np.linalg.norm(
            left_mouth - nose
        )

        right_mouth_nose = np.linalg.norm(
            right_mouth - nose
        )

        # ---------------------
        # Symmetry Scores
        # ---------------------
        eye_ratio = (
            min(
                left_eye_nose,
                right_eye_nose
            )
            /
            max(
                left_eye_nose,
                right_eye_nose
            )
        )

        mouth_ratio = (
            min(
                left_mouth_nose,
                right_mouth_nose
            )
            /
            max(
                left_mouth_nose,
                right_mouth_nose
            )
        )

        frontal_score = (
            eye_ratio +
            mouth_ratio
        ) / 2

        # ---------------------
        # Face Size
        # ---------------------
        eye_distance = np.linalg.norm(
            left_eye - right_eye
        )

        # ---------------------
        # Final Quality
        # ---------------------
        quality_score = (
            frontal_score *
            face.det_score *
            eye_distance
        )

        

        # ---------------------
        # Find Matching Track
        # ---------------------
        face_center_x = (
            fx1 + fx2
        ) // 2

        face_center_y = (
            fy1 + fy2
        ) // 2

        matched_track = None

        for bbox, track_id in zip(
            detections.xyxy,
            detections.tracker_id
        ):

            if track_id is None:
                continue

            px1, py1, px2, py2 = map(
                int,
                bbox
            )

            if (
                px1 <= face_center_x <= px2
                and
                py1 <= face_center_y <= py2
            ):

                matched_track = track_id
                break

        # ---------------------
        # Store Best Face
        # ---------------------
        if matched_track is not None:

            if (
                matched_track not in best_face_quality
            ):

                best_face_quality[
                    matched_track
                ] = quality_score

                best_face_embedding[
                    matched_track
                ] = face.embedding
                face_crop = frame[
                    fy1:fy2,
                    fx1:fx2
                ].copy()

                best_face_crop[
                    matched_track
                ] = face_crop
            elif (
                quality_score >
                best_face_quality[
                    matched_track
                ]
            ):

                best_face_quality[
                    matched_track
                ] = quality_score

                best_face_embedding[
                    matched_track
                ] = face.embedding
                face_crop = frame[
                    fy1:fy2,
                    fx1:fx2
                ].copy()

                best_face_crop[
                    matched_track
                ] = face_crop
                print(
                    f"Track {matched_track}"
                    f" New Best Quality:"
                    f" {quality_score:.2f}"
                )
        cv2.rectangle(
            frame,
            (fx1, fy1),
            (fx2, fy2),
            (0, 255, 0),
            2
        )
        cv2.putText(
            frame,
            f"Q:{quality_score:.1f} F:{frontal_score:.2f}",
            (fx1, fy1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
    cv2.imshow(
        "Integration Test",
        frame
    )
    ended_tracks = (
        active_tracks -
        current_tracks
    )
    for track_id in ended_tracks:

        print(
            f"\nTrack {track_id} ended"
        )

        if track_id not in best_face_embedding:
            continue

        query_embedding = best_face_embedding[
            track_id
        ]

        # ---------------------
        # Search Database
        # ---------------------
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

        # ---------------------
        # Existing Person
        # ---------------------
        if best_score > THRESHOLD:

            print(
                f"Track {track_id}"
                f" -> Existing Person"
            )

            print(
                f"Person ID: {best_id}"
            )

            print(
                f"Similarity: {best_score:.4f}"
            )

        # ---------------------
        # New Person
        # ---------------------
        else:

            embedding_blob = pickle.dumps(
                query_embedding
            )

            cursor.execute(
                """
                INSERT INTO persons
                (embedding)
                VALUES (?)
                """,
                (embedding_blob,)
            )

            conn.commit()

            new_id = cursor.lastrowid

            print(
                f"Track {track_id}"
                f" -> Registered"
            )

            print(
                f"Assigned ID: {new_id}"
            )

        # ---------------------
        # Save Face Image
        # ---------------------
        if track_id in best_face_crop:

            cv2.imwrite(
                f"track_{track_id}.jpg",
                best_face_crop[track_id]
            )
    active_tracks = current_tracks.copy()
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
conn.close()