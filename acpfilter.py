import cv2
import mediapipe as mp
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

filters = ["Normal", "Grayscale", "Sepia", "Negative", "Blur"]
filter_index = 0
last_action = time.time()

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            h, w, c = frame.shape

            thumb = hand.landmark[4]
            index = hand.landmark[8]
            middle = hand.landmark[12]

            tx, ty = int(thumb.x * w), int(thumb.y * h)
            ix, iy = int(index.x * w), int(index.y * h)
            mx, my = int(middle.x * w), int(middle.y * h)

            dist_index = math.hypot(ix - tx, iy - ty)
            dist_middle = math.hypot(mx - tx, my - ty)

            current_time = time.time()

            # Capture photo (Thumb + Index)
            if dist_index < 30 and current_time - last_action > 1:
                cv2.imwrite("captured_photo.jpg", frame)
                print("Photo Captured!")
                last_action = current_time

            # Change filter (Thumb + Middle)
            elif dist_middle < 30 and current_time - last_action > 1:
                filter_index = (filter_index + 1) % len(filters)
                print("Filter:", filters[filter_index])
                last_action = current_time

    # Apply filters
    if filters[filter_index] == "Grayscale":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    elif filters[filter_index] == "Sepia":
        kernel = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])
        frame = cv2.transform(frame, kernel)
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    elif filters[filter_index] == "Negative":
        frame = cv2.bitwise_not(frame)

    elif filters[filter_index] == "Blur":
        frame = cv2.GaussianBlur(frame, (15, 15), 0)

    cv2.putText(
        frame,
        f"Filter: {filters[filter_index]}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Gesture Controlled Photo App", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()