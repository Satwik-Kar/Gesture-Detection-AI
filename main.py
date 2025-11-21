import cv2
import numpy as np
import mediapipe as mp
import time

width = 1920
height = 1080
window_name = "Hand Gesture App"

min_confidence = 0.8
brush_size = 15
eraser_size = 150
hover_radius = 15

blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)
yellow = (0, 255, 255)
white = (255, 255, 255)
black = (0, 0, 0)
header_bg = (40, 40, 40)
slider_bg = (50, 50, 50)

active_color = red

slider_x_min = 800  
slider_x_max = width - 100
slider_y_min = height // 2 - 50
slider_y_max = height // 2 + 50

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=min_confidence, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

canvas = None
mode = "Writing"
xp, yp = 0, 0
t_last_trigger = 0

def count_fingers(landmarks):
    thumb = landmarks[4]
    thumb_ip = landmarks[3]
    pinky_mcp = landmarks[17]

    if thumb.x > pinky_mcp.x:
        thumb_up = thumb.x > thumb_ip.x + 0.02
    else:
        thumb_up = thumb.x < thumb_ip.x - 0.02

    index_up = landmarks[8].y < landmarks[6].y
    middle_up = landmarks[12].y < landmarks[10].y
    ring_up = landmarks[16].y < landmarks[14].y
    pinky_up = landmarks[20].y < landmarks[18].y

    return [thumb_up, index_up, middle_up, ring_up, pinky_up]

def detect_gesture(fingers, landmarks):
    thumb, index, middle, ring, pinky = fingers
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]

    if all(fingers): return "Open Hand"
    elif not any(fingers): return "Fist"
    elif thumb and not any(fingers[1:]):
        return "Thumbs Up" if thumb_tip.y < thumb_ip.y else "Thumbs Down"
    elif index and middle and not any(fingers[3:]): return "Peace"
    elif thumb and index and not any(fingers[2:]): return "L Sign"
    elif index and pinky and not any([middle, ring]):
        return "Spiderman" if thumb else "Rock!!"
    elif thumb and pinky and not any([index, middle, ring]): return "Call Me"
    elif index and middle and ring and not pinky: return "Three"
    elif index and middle and ring and pinky and not thumb: return "Four"
    elif index and not any([middle, ring]): return "Pointed"
    elif middle and ring and pinky and not any([index, thumb]): return "Perfect!!"
    else: return "Unknown"

def draw_ui(img):
    cv2.rectangle(img, (300, 10), (420, 90), blue, cv2.FILLED)
    cv2.rectangle(img, (440, 10), (560, 90), green, cv2.FILLED)
    cv2.rectangle(img, (580, 10), (700, 90), red, cv2.FILLED)
    cv2.rectangle(img, (720, 10), (840, 90), yellow, cv2.FILLED)
    
    cv2.rectangle(img, (300, 10), (420, 90), white, 2)
    cv2.rectangle(img, (440, 10), (560, 90), white, 2)
    cv2.rectangle(img, (580, 10), (700, 90), white, 2)
    cv2.rectangle(img, (720, 10), (840, 90), white, 2)

while True:
    success, frame = cap.read()
    if not success: break

    frame = cv2.resize(frame, (width, height))
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    if canvas is None:
        canvas = np.zeros((h, w, 3), np.uint8)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    cv2.rectangle(frame, (0, 0), (w, 100), header_bg, cv2.FILLED)

    slider_mode = False
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            cx = hand_lms.landmark[9].x 
            fingers = count_fingers(hand_lms.landmark)
            
            if cx < 0.5 and all(fingers):
                slider_mode = True
                t_last_trigger = time.time()
                break 

    if mode == "Writing":
        draw_ui(frame)
        
        cv2.rectangle(frame, (width - 160, 10), (width - 10, 90), white, cv2.FILLED)
        cv2.putText(frame, f"{brush_size}", (width - 100, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, black, 5)
        cv2.circle(frame, (width - 130, 50), 15, active_color, cv2.FILLED)
        
        if slider_mode:
            cv2.rectangle(frame, (slider_x_min, slider_y_min), (slider_x_max, slider_y_max), slider_bg, cv2.FILLED)
            cv2.rectangle(frame, (slider_x_min, slider_y_min), (slider_x_max, slider_y_max), white, 3)
            
            bar_width = int(np.interp(brush_size, [5, 100], [slider_x_min, slider_x_max]))
            cv2.rectangle(frame, (slider_x_min, slider_y_min), (bar_width, slider_y_max), active_color, cv2.FILLED)
            
            cv2.putText(frame, "Slide Right Hand", (slider_x_min, slider_y_max + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, white, 2)
        
        elif time.time() - t_last_trigger < 1.0:
            cv2.putText(frame, "Ready...", (w//2 - 100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, white, 3)
            xp, yp = 0, 0

        else:
            instruction_text = "1 Finger: Draw | Left Open: Size Menu | 2 Fingers: Select Color | 3 Fingers: Erase"
            cv2.putText(frame, instruction_text, (width//2 - 500, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    else:
        cv2.putText(frame, f"Mode: {mode}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, white, 2)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            fingers = count_fingers(hand_lms.landmark)
            cx = hand_lms.landmark[9].x
            
            lm_list = []
            for id, lm in enumerate(hand_lms.landmark):
                px, py = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, px, py])

            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

            if len(lm_list) != 0:
                x1, y1 = lm_list[8][1:]

                if slider_mode and cx < 0.5:
                    continue

                if mode == "Gesture":
                    g_name = detect_gesture(fingers, hand_lms.landmark)
                    cv2.putText(frame, g_name, (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, yellow, 3)
                    xp, yp = 0, 0

                elif mode == "Writing":
                    
                    if slider_mode:
                        if fingers[1] and not fingers[2]: 
                             if slider_y_min - 50 < y1 < slider_y_max + 50:
                                 brush_size = int(np.interp(x1, [slider_x_min, slider_x_max], [5, 100]))
                                 if brush_size < 5: brush_size = 5
                                 if brush_size > 100: brush_size = 100
                                 cv2.circle(frame, (x1, y1), 20, white, cv2.FILLED)
                        xp, yp = 0, 0 

                    elif time.time() - t_last_trigger < 1.0:
                        xp, yp = 0, 0

                    else:
                        if fingers[1] and fingers[2] and fingers[3]:
                            cv2.circle(frame, (x1, y1), eraser_size // 2, white, cv2.FILLED)
                            if xp == 0 and yp == 0: xp, yp = x1, y1
                            cv2.line(canvas, (xp, yp), (x1, y1), black, eraser_size)
                            xp, yp = x1, y1

                        elif fingers[1] and fingers[2]:
                            xp, yp = x1, y1
                            if y1 < 100:
                                if 300 < x1 < 420: active_color = blue
                                elif 440 < x1 < 560: active_color = green
                                elif 580 < x1 < 700: active_color = red
                                elif 720 < x1 < 840: active_color = yellow
                                cv2.rectangle(frame, (x1-15, y1-15), (x1+15, y1+15), white, 2)
                            else:
                                cv2.circle(frame, (x1, y1), hover_radius, active_color, 2)

                        elif fingers[1] and not fingers[2]:
                            if y1 < 100:
                                xp, yp = 0, 0
                            else:
                                cv2.circle(frame, (x1, y1), int(brush_size/2), active_color, cv2.FILLED)
                                if xp == 0 and yp == 0: xp, yp = x1, y1
                                cv2.line(canvas, (xp, yp), (x1, y1), active_color, brush_size)
                                xp, yp = x1, y1
                        
                        else:
                            xp, yp = 0, 0
    else:
        xp, yp = 0, 0

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, inv)
    frame = cv2.bitwise_or(frame, canvas)

    cv2.imshow(window_name, frame)
    
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27: break
    elif key == ord('c'): canvas = np.zeros((h, w, 3), np.uint8)
    elif key == ord('m'): mode = "Gesture" if mode == "Writing" else "Writing"

cap.release()
cv2.destroyAllWindows()