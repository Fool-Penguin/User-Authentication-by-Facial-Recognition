# face_service.py
from deepface import DeepFace
from ultralytics import YOLO
from datetime import datetime
import pandas as pd
import numpy as np
import json
import time
import cv2
import csv
import os

# ----------------- Logging -----------------
def log_access(person_name, is_real, authorized,
               face_image=None,
               log_path="access_log.csv",
               save_faces=True):

    # Ensure folder for CSV (current folder if no directory part)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    file_exists = os.path.exists(log_path)

    with open(log_path, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Header if new file
        if not file_exists:
            writer.writerow(["Timestamp", "Person", "Spoofing", "Authorized", "Face_Image"])

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        spoof_status = "Real" if is_real else "Fake"
        auth_status = "Yes" if authorized else "No"
        image_filename = ""

        if save_faces and face_image is not None:
            folder = "Authorized" if authorized else "Unauthorized"
            os.makedirs(folder, exist_ok=True)

            clean_name = (person_name or "Unknown").replace(" ", "_")
            image_filename = os.path.join(folder, f"{timestamp}_{clean_name}.jpg")

            cv2.imwrite(image_filename, face_image)

        writer.writerow([timestamp, person_name, spoof_status, auth_status, image_filename])

    print(f"[LOG] {person_name} | Spoofing: {spoof_status} | Authorized: {auth_status}")


# ----------------- Model loading -----------------
def load_model():
    # Make sure yolov11n-face.pt is in the same folder
    face_detector = YOLO("yolov11n-face.pt")
    face_recognizer = DeepFace
    return face_detector, face_recognizer


face_detector, face_recognizer = load_model()
print("Models loaded (for web API).")


# ----------------- Face detection -----------------
def detect_face(frame):
    """
    frame: BGR image (numpy array)
    returns: (cropped_face, num_faces)
    """
    results = face_detector(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    num_faces = len(boxes)
    if num_faces == 0:
        return None, 0

    # Use first detected face
    x1, y1, x2, y2 = map(int, boxes[0])

    h, w = frame.shape[:2]
    y1 = max(0, y1 - 75)
    x1 = max(0, x1 - 75)
    y2 = min(h, y2 + 75)
    x2 = min(w, x2 + 75)

    cropped_face = frame[y1:y2, x1:x2]
    return cropped_face, num_faces


# ----------------- Recognition -----------------
def recognize_face(input_, database):
    """
    input_: BGR face image
    database: path to faceDB folder
    returns: (found_bool, record_dict_or_None)
    """
    try:
        result = DeepFace.find(input_, database, enforce_detection=False)
        print(str(result))

        if not result or len(result[0]) == 0:
            return False, None

        df = pd.DataFrame(result[0])
        record = json.loads(df.to_json(orient="records"))

        with open("match.json", "w") as f:
            for i in record:
                try:
                    if i.get('confidence', 0) >= 65:
                        f.write(json.dumps(i["identity"].strip("faceDB/"), indent=2))
                        f.write("\n")
                except Exception as e:
                    print("Error writing identity: ", str(e))
                    return False, None
            print("JSON File Saved")

        # Best match = first row
        return True, record[0]

    except Exception as e:
        print("Some errors occurred: ", str(e))
        return False, None


# ----------------- Main helper for web -----------------
def process_image_bytes(image_bytes, database="faceDB"):
    """
    Main function for Flask.
    image_bytes: raw bytes from uploaded file
    returns: dict for JSON response
    """

    start_time = time.time()

    # 1) Decode bytes -> BGR image
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {
            "ok": False,
            "recognized": False,
            "real": None,
            "authorized": False,
            "name": None,
            "reason": "decode_failed",
            "time_ms": 0,
        }

    # 2) Detect face(s)
    face, face_num = detect_face(frame)
    if face is None or face_num == 0:
        return {
            "ok": True,
            "recognized": False,
            "real": None,
            "authorized": False,
            "name": None,
            "reason": "no_face_detected",
            "time_ms": int((time.time() - start_time) * 1000),
        }

    if face_num > 1:
        return {
            "ok": False,
            "recognized": False,
            "real": None,
            "authorized": False,
            "name": None,
            "reason": "multiple_faces_detected",
            "time_ms": int((time.time() - start_time) * 1000),
        }

    # 3) Anti-spoofing
    spoof_check = DeepFace.extract_faces(face, anti_spoofing=True, enforce_detection=False)
    real = spoof_check[0].get("is_real", False)
    print(f"Spoofing check result: {real}")

    # 4) Recognition
    found, who = recognize_face(face, database)
    who_name = "Unknown"
    authorized = False

    if found and who is not None:
        identity = who.get("identity", "")
        base = os.path.basename(identity)          # e.g. Ongsa1.jpg
        who_name = os.path.splitext(base)[0]       # -> Ongsa1
        print(f"Found face in database: {who_name}")

        if real:
            authorized = True
            print(f"Found face in database! Welcome back, {who_name} 😊🙏🥀")
        else:
            authorized = False
            print("Spoofing detected for a known person!")
    else:
        print("Not found in database.")

    # 5) Log
    log_access(
        person_name=who_name if found else "Unknown",
        is_real=real,
        authorized=authorized,
        face_image=face
    )

    end_time = time.time()

    return {
        "ok": True,
        "recognized": bool(found),
        "real": bool(real),
        "authorized": bool(authorized),
        "name": who_name if found else None,
        "reason": "ok" if authorized or found else "not_in_db_or_spoof",
        "time_ms": int((end_time - start_time) * 1000),
    }
