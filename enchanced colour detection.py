import cv2
import numpy as np

# Start webcam
cap = cv2.VideoCapture(0)

mode = "original"

print("Controls:")
print("r - Red Filter")
print("g - Green Filter")
print("b - Blue Filter")
print("c - Canny Edge Detection")
print("s - Sobel Edge Detection")
print("l - Laplacian Edge Detection")
print("o - Original View")
print("q - Quit")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    output = frame.copy()

    # Apply selected mode
    if mode == "red":
        output[:, :, 0] = 0
        output[:, :, 1] = 0

    elif mode == "green":
        output[:, :, 0] = 0
        output[:, :, 2] = 0

    elif mode == "blue":
        output[:, :, 1] = 0
        output[:, :, 2] = 0

    elif mode == "canny":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        output = cv2.Canny(gray, 100, 200)

    elif mode == "sobel":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        output = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)

    elif mode == "laplacian":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        output = cv2.Laplacian(gray, cv2.CV_64F)

    cv2.imshow("Image Processing", output)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        mode = "red"
    elif key == ord('g'):
        mode = "green"
    elif key == ord('b'):
        mode = "blue"
    elif key == ord('c'):
        mode = "canny"
    elif key == ord('s'):
        mode = "sobel"
    elif key == ord('l'):
        mode = "laplacian"
    elif key == ord('o'):
        mode = "original"
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()