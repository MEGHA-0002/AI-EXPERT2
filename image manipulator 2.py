import cv2
import numpy as np

# Load image
image = cv2.imread("input.jpg")

if image is None:
    print("Error: Image not found!")
    exit()

# 1. Rotate the image by 45 degrees
height, width = image.shape[:2]
center = (width // 2, height // 2)

rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(image, rotation_matrix, (width, height))

# 2. Crop the center region
cropped = image[100:400, 100:400]

# 3. Increase brightness
brightened = cv2.convertScaleAbs(image, alpha=1.0, beta=50)

# Display results
cv2.imshow("Original Image", image)
cv2.imshow("Rotated Image", rotated)
cv2.imshow("Cropped Image", cropped)
cv2.imshow("Brightened Image", brightened)

# Save manipulated images
cv2.imwrite("rotated_image.jpg", rotated)
cv2.imwrite("cropped_image.jpg", cropped)
cv2.imwrite("brightened_image.jpg", brightened)

print("Images processed and saved successfully!")

cv2.waitKey(0)
cv2.destroyAllWindows()