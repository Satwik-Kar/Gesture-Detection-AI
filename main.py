import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Open the webcam
cap = cv2.VideoCapture(0)

def classify_hand_landmarks(landmarks):
    # Check if thumb is open
    thumb_open = landmarks[4].x > landmarks[3].x if landmarks[4].x > landmarks[17].x else landmarks[4].x < landmarks[3].x
    # Check if other fingers are open
    index_finger_open = landmarks[8].y < landmarks[6].y
    middle_finger_open = landmarks[12].y < landmarks[10].y
    ring_finger_open = landmarks[16].y < landmarks[14].y
    pinky_finger_open = landmarks[20].y < landmarks[18].y

    # Determine gestures
    if all([thumb_open, index_finger_open, middle_finger_open, ring_finger_open, pinky_finger_open]):
        return "Open Hand"
    elif not any([thumb_open, index_finger_open, middle_finger_open, ring_finger_open, pinky_finger_open]):
        return "Fist"
    elif thumb_open and not any([index_finger_open, middle_finger_open, ring_finger_open, pinky_finger_open]):
        return "Thumbs Up" if landmarks[4].y < landmarks[3].y else "Thumbs Down"
    elif index_finger_open and middle_finger_open and not any([ring_finger_open, pinky_finger_open]):
        return "Peace"
    elif thumb_open and index_finger_open and not any([middle_finger_open, ring_finger_open, pinky_finger_open]):
        return "Okay"
    elif index_finger_open and pinky_finger_open and not any([middle_finger_open, ring_finger_open]):
        return "Rock!!"
    elif index_finger_open and not any([middle_finger_open, ring_finger_open]):
        return "Pointed"
    elif middle_finger_open and ring_finger_open and pinky_finger_open and not any([index_finger_open, thumb_open]):
        return "Perfect!!"
    else:
        return "Unknown"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and find hands
    results = hands.process(frame_rgb)

    # Draw hand landmarks and classify gestures
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Convert landmarks to a list of tuples
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

            # Classify the gesture
            gesture_name = classify_hand_landmarks(hand_landmarks.landmark)

            # Display the gesture name on the frame
            cv2.putText(frame, gesture_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Hand Gesture Detection', frame)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
