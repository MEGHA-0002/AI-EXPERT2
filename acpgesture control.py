import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            # Thumb tip = 4, Index tip = 8
            x1, y1 = int(handLms.landmark[4].x * img.shape[1]), int(handLms.landmark[4].y * img.shape[0])
            x2, y2 = int(handLms.landmark[8].x * img.shape[1]), int(handLms.landmark[8].y * img.shape[0])

            # Draw points and line
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), -1)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), -1)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Distance between thumb and index finger
            length = np.hypot(x2 - x1, y2 - y1)

            # Convert distance to percentage
            percent = np.interp(length, [20, 200], [0, 100])

            cv2.putText(
                img,
                f'Control: {int(percent)}%',
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            mp_draw.draw_landmarks(
                img,
                handLms,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("Gesture Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()