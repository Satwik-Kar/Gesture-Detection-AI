import cv2
import numpy as np
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

window_name = "Hand Gesture & Writing App"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

img_canvas = None
brush_thickness = 10
eraser_thickness = 150
draw_color = (255, 0, 255)
brush_radius = 130
current_mode = "Writing"

xp, yp = 0, 0

def classify_hand_landmarks(landmarks):
    thumb_open = landmarks[4].x > landmarks[3].x if landmarks[4].x > landmarks[17].x else landmarks[4].x < landmarks[3].x
    index_finger_open = landmarks[8].y < landmarks[6].y
    middle_finger_open = landmarks[12].y < landmarks[10].y
    ring_finger_open = landmarks[16].y < landmarks[14].y
    pinky_finger_open = landmarks[20].y < landmarks[18].y

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

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.resize(img, (1920, 1080))
    img = cv2.flip(img, 1)
    h, w, c = img.shape

    if img_canvas is None:
        img_canvas = np.zeros((h, w, 3), np.uint8)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    cv2.rectangle(img, (0, 0), (w, 100), (40, 40, 40), cv2.FILLED)
    
    cv2.putText(img, f"Mode (Press 'm'): {current_mode}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if current_mode == "Writing":
        cv2.putText(img, "1 Finger: Write  |  2 Fingers: Hover  |  3 Fingers: Erase", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_lms.landmark):
                px, py = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, px, py])

            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

            if len(lm_list) != 0:
                if current_mode == "Gesture":
                    gesture_name = classify_hand_landmarks(hand_lms.landmark)
                    cv2.putText(img, f"Gesture: {gesture_name}", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
                    xp, yp = 0, 0

                elif current_mode == "Writing":
                    x1, y1 = lm_list[8][1:]
                    
                    fingers = []
                    if lm_list[4][1] < lm_list[3][1]: 
                        fingers.append(1)
                    else:
                        fingers.append(0)

                    for id in range(8, 21, 4):
                        if lm_list[id][2] < lm_list[id - 2][2]:
                            fingers.append(1)
                        else:
                            fingers.append(0)

                    if fingers[1] and fingers[2] and fingers[3]:
                        cv2.circle(img, (x1, y1), brush_radius, (0, 0, 0), cv2.FILLED)
                        cv2.putText(img, "Eraser", (x1+35, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        
                        if xp == 0 and yp == 0:
                            xp, yp = x1, y1
                        
                        cv2.line(img_canvas, (xp, yp), (x1, y1), (0, 0, 0), eraser_thickness)
                        xp, yp = x1, y1

                    elif fingers[1] and fingers[2]:
                        cv2.circle(img, (x1, y1), 15, (0, 255, 0), 2)
                        xp, yp = x1, y1

                    elif fingers[1] and not fingers[2]:
                        cv2.circle(img, (x1, y1), 15, draw_color, cv2.FILLED)
                        
                        if xp == 0 and yp == 0:
                            xp, yp = x1, y1
                        
                        cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, brush_thickness)
                        xp, yp = x1, y1
                    else:
                        xp, yp = 0, 0
    else:
        xp, yp = 0, 0

    img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, img_canvas)

    cv2.imshow(window_name, img)
    
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        break
    elif key == ord('c'):
        img_canvas = np.zeros((h, w, 3), np.uint8)
    elif key == ord('m'):
        if current_mode == "Writing":
            current_mode = "Gesture"
        else:
            current_mode = "Writing"

cap.release()
cv2.destroyAllWindows()