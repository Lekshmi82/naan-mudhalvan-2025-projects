import cv2
import numpy as np

# Load the image
image = cv2.imread("underwater_sample.jpg")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define yellow color range (tuned for fish)
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([35, 255, 255])

# Mask and clean noise
mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Find contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Only keep the largest contour assuming it's the fish
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, "Fish", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Show result
cv2.imshow("Fish Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()




