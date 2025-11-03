from ultralytics import YOLO
import cv2
import numpy as np
from deepface import DeepFace
from keras.models import model_from_yaml
from keras.preprocessing.image import img_to_array

def load_models():
    yaml_file = open("trained_model/RGB_rPPG_merge_softmax_.yaml", 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    liveness_model = model_from_yaml(loaded_model_yaml)

    print("Loading models (detector, liveness, recognizer)...")
    
    face_detector = YOLO('yolov11n-face.pt')

    liveness_model.load_weights("trained_model/RGB_rPPG_merge_softmax_.h5")

    face_recognizer = YOLO("yolo11freeze.pt")
    
    # For this skeleton, we'll return placeholders
    return face_detector, liveness_model, face_recognizer

def detect_faces(frame, detector_model):
    results = detector_model(frame)
    boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2))
    return boxes    


def check_liveness(face_crop, liveness_model):

    result = liveness_model.run(None, {"input": face_crop})
    score = result[0][1]  # Assuming index 1 is the 'live
    is_live = score > 0.5  # Example threshold
    return is_live, score

def get_recognition_result(face_crop, recognizer_model, known_db):
    """
    Takes a live face crop, the recognizer, and the known user database.
    Returns the name of the recognized user or "Unknown".
    """
    # --- This is where you would:
    # 1. Pre-process the face_crop
    # 2. Run the recognizer_model to get a 512-d embedding (vector)
    # 3. Compare this vector to all vectors in your 'known_db'
    # 4. Find the closest match using cosine similarity
    
    # Placeholder: Return a mock result
    # In a real app, 'known_db' would be a dict like:
    # {"Alice": [vector1, vector2], "Bob": [vector3]}
    
    user_name = "Alice" # Assume we found a match
    
    # if no_match_found:
    #    user_name = "Unknown"
        
    return user_name

# --- 2. Main Application Logic ---

def main():
    # Load your models once at the start
    face_detector, liveness_model, recognizer = load_models()
    
    # Load your database of known users (you must create this)
    # This involves running your 'recognizer' on 5-10 photos
    # of each allowed user and saving their vectors.
    known_user_db = {"Mark": [], "Peem": [], "Ongsa": []} 
    print("Known user database loaded.")

    # Start the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # --- Step 1: Face Detection ---
        # Find all faces in the current frame
        face_boxes = detect_faces(frame, face_detector)

        for (x1, y1, x2, y2) in face_boxes:
            # Crop the face from the frame
            # Add some padding (e.g., 10px) if needed
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                continue

            # --- Step 2: Liveness Detection ---
            is_live, liveness_score = check_liveness(face_crop, liveness_model)

            if not is_live:
                # SPOOF DETECTED!
                color = (0, 0, 255)  # Red
                text = f"SPOOF DETECTED ({liveness_score:.2f})"
                
            else:
                # LIVE! Now we can check who it is.
                # --- Step 3: Face Recognition ---
                user_name = get_recognition_result(face_crop, recognizer, known_user_db)
                
                if user_name == "Unknown":
                    color = (0, 255, 255) # Yellow
                    text = "Unknown User"
                else:
                    color = (0, 255, 0) # Green
                    text = f"Welcome, {user_name}!"

            # Draw the result on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Display the final frame
        cv2.imshow("Face Recognition Security", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()