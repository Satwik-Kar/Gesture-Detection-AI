import cv2
import numpy as np
import mediapipe as mp

CAM_WIDTH = 1920
CAM_HEIGHT = 1080
WINDOW_NAME = "Hand Gesture & Writing App"

MIN_DETECTION_CONF = 0.8
MIN_TRACKING_CONF = 0.8

BRUSH_THICKNESS = 10
ERASER_SIZE = 150 
HOVER_INDICATOR_RADIUS = 15

COLOR_DRAW = (0, 0, 255)      
COLOR_ERASER = (255, 255, 255)        
COLOR_HOVER = (0, 255, 0)       
COLOR_HEADER_BG = (40, 40, 40)  
COLOR_TEXT_MAIN = (255, 255, 255)
COLOR_TEXT_SUB = (200, 200, 200)
COLOR_TEXT_GESTURE = (255, 255, 0)

KEY_QUIT_Q = ord('q')
KEY_QUIT_ESC = 27
KEY_CLEAR = ord('c')
KEY_MODE = ord('m')

MODE_WRITING = "Writing"
MODE_GESTURE = "Gesture"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=MIN_DETECTION_CONF, 
    min_tracking_confidence=MIN_TRACKING_CONF, 
    max_num_hands=1
)
mp_draw = mp.solutions.drawing_utils

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cap = cv2.VideoCapture(0)
cap.set(3, CAM_WIDTH)
cap.set(4, CAM_HEIGHT)

img_canvas = None
current_mode = MODE_WRITING
xp, yp = 0, 0

def get_fingers_status(landmarks):
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    pinky_mcp = landmarks[17]

    if thumb_tip.x > pinky_mcp.x:
        thumb_open = thumb_tip.x > thumb_ip.x
    else:
        thumb_open = thumb_tip.x < thumb_ip.x

    index_open = landmarks[8].y < landmarks[6].y
    middle_open = landmarks[12].y < landmarks[10].y
    ring_open = landmarks[16].y < landmarks[14].y
    pinky_open = landmarks[20].y < landmarks[18].y

    return [thumb_open, index_open, middle_open, ring_open, pinky_open]

def classify_hand_landmarks(landmarks, fingers):
    thumb_open, index_open, middle_open, ring_open, pinky_open = fingers
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]

    if all(fingers):
        return "Open Hand"
    elif not any(fingers):
        return "Fist"
    elif thumb_open and not any(fingers[1:]):
        if thumb_tip.y < thumb_ip.y:
            return "Thumbs Up"
        else:
            return "Thumbs Down"
    elif index_open and middle_open and not any(fingers[3:]):
        return "Peace"
    elif thumb_open and index_open and not any(fingers[2:]):
        return "L Sign"
    elif index_open and pinky_open and not any([middle_open, ring_open]):
        if thumb_open:
            return "Spiderman"
        else:
            return "Rock!!"
    elif thumb_open and pinky_open and not any([index_open, middle_open, ring_open]):
        return "Call Me"
    elif index_open and middle_open and ring_open and not pinky_open:
        return "Three"
    elif index_open and middle_open and ring_open and pinky_open and not thumb_open:
        return "Four"
    elif index_open and not any([middle_open, ring_open]):
        return "Pointed"
    elif middle_open and ring_open and pinky_open and not any([index_open, thumb_open]):
        return "Perfect!!"
    else:
        return "Unknown"

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.resize(img, (CAM_WIDTH, CAM_HEIGHT))
    img = cv2.flip(img, 1)
    h, w, c = img.shape

    if img_canvas is None:
        img_canvas = np.zeros((h, w, 3), np.uint8)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    cv2.rectangle(img, (0, 0), (w, 100), COLOR_HEADER_BG, cv2.FILLED)
    cv2.putText(img, f"Mode (Press 'm'): {current_mode}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_TEXT_MAIN, 2)

    if current_mode == MODE_WRITING:
        cv2.putText(img, "1 Finger: Write  |  2 Fingers: Hover  |  3 Fingers: Erase", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT_SUB, 2)
    
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            fingers = get_fingers_status(hand_lms.landmark)
            
            lm_list = []
            for id, lm in enumerate(hand_lms.landmark):
                px, py = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, px, py])

            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

            if len(lm_list) != 0:
                if current_mode == MODE_GESTURE:
                    gesture_name = classify_hand_landmarks(hand_lms.landmark, fingers)
                    cv2.putText(img, f"Gesture: {gesture_name}", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_TEXT_GESTURE, 3)
                    xp, yp = 0, 0

                elif current_mode == MODE_WRITING:
                    x1, y1 = lm_list[8][1:]
                    
                    if fingers[1] and fingers[2] and fingers[3]:
                        cv2.circle(img, (x1, y1), ERASER_SIZE // 2, COLOR_ERASER, cv2.FILLED)
                        cv2.putText(img, "Eraser", (x1+35, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ERASER, 1)
                        
                        if xp == 0 and yp == 0:
                            xp, yp = x1, y1
                        
                        cv2.line(img_canvas, (xp, yp), (x1, y1), (0, 0, 0), ERASER_SIZE)
                        xp, yp = x1, y1

                    elif fingers[1] and fingers[2]:
                        cv2.circle(img, (x1, y1), HOVER_INDICATOR_RADIUS, COLOR_HOVER, 2)
                        xp, yp = x1, y1

                    elif fingers[1] and not fingers[2]:
                        cv2.circle(img, (x1, y1), HOVER_INDICATOR_RADIUS, COLOR_DRAW, cv2.FILLED)
                        
                        if xp == 0 and yp == 0:
                            xp, yp = x1, y1
                        
                        cv2.line(img_canvas, (xp, yp), (x1, y1), COLOR_DRAW, BRUSH_THICKNESS)
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

    cv2.imshow(WINDOW_NAME, img)
    
    key = cv2.waitKey(1)
    if key == KEY_QUIT_Q or key == KEY_QUIT_ESC:
        break
    elif key == KEY_CLEAR:
        img_canvas = np.zeros((h, w, 3), np.uint8)
    elif key == KEY_MODE:
        if current_mode == MODE_WRITING:
            current_mode = MODE_GESTURE
        else:
            current_mode = MODE_WRITING

cap.release()
cv2.destroyAllWindows()