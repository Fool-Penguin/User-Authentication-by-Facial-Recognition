from ultralytics import YOLO
import cv2
import numpy as np
from deepface import DeepFace
import json
import pandas as pd
import re
import time
def load_model():
    face_datector = YOLO("yolov11n-face.pt")
    face_regonizer = DeepFace
    return face_datector, face_regonizer
def Init_Camera():
    cap = cv2.VideoCapture(0)
    print("Camera is starting...")
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    return cap
def get_picture(camera):
    print("Getting frame from camera...")
    ret, frame = camera.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        return None
    return frame
def close_Camera(camera):
    camera.release()
    cv2.destroyAllWindows()

def detect_face(face_datector, frame):
    results = face_datector(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Extract bounding box coordinates
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
        cropped_face = frame[y1-50:y2+50, x1-50:x2+50]  # Crop the face from the frame
    return cropped_face
def regonition_face():
    return 0
def main():
    starttime = time.time()
    face_datector, face_regonizer = load_model()
    print("Models loaded.")
    print("Initializing camera...")
    camera = Init_Camera()
    try:
        image = get_picture(camera)

        if image is not None:
            cv2.imwrite("picture_1.jpg", image)
            print(f"Picture 1 saved. Image shape: {image.shape}")
            results = detect_face(face_datector, image)
            cv2.imwrite("cropped_face.jpg", results)
            print(f"Cropped face saved. Cropped shape: {results.shape}")

    except SystemExit as e:
        print(e)       
    finally:

        print("Releasing camera...")
        close_Camera(camera)
    endtime = time.time()
    print(f"Total time taken: {endtime - starttime} seconds")


if __name__ == "__main__":
    main()