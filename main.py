import cv2
import mediapipe as mp
import sys

print("Step 1: Initializing MediaPipe...")
mp_face_mesh = mp.solutions.face_mesh

# Add these specific parameters to make it initialize faster/easier
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
print("Step 1.5: MediaPipe Initialized Successfully!")

print("Step 2: Accessing Webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam. Try changing index 0 to 1.")
    sys.exit()

print("Step 3: Entering Loop (Press 'Esc' to quit)...")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Simple text overlay to prove it's working
    cv2.putText(image, "System Active", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Driver Monitor', image)
    
    # Wait for 1ms and check for ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

print("Step 4: Cleaning up...")
cap.release()
cv2.destroyAllWindows()