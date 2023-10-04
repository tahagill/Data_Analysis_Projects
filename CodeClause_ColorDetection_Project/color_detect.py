import numpy as np
import cv2

def get_yellow_hsv_limits():
    # Define the HSV range for yellow color
    lower_limit = np.array([20, 100, 100], dtype=np.uint8)
    upper_limit = np.array([30, 255, 255], dtype=np.uint8)
    return lower_limit, upper_limit

def detect_and_highlight_yellow(frame):
    # Convert the frame to the HSV color space
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get the lower and upper limits for yellow color
    lower_limit, upper_limit = get_yellow_hsv_limits()

    # Create a mask to extract yellow regions
    mask = cv2.inRange(hsv_image, lower_limit, upper_limit)

    # Find contours of yellow regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around detected yellow regions
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)

yellow = [0, 255, 255]  # Yellow in BGR colorspace
cap = cv2.VideoCapture(0)  # Use 0 for the default camera or change to a specific camera index

while True:
    ret, frame = cap.read()

    detect_and_highlight_yellow(frame)

    cv2.imshow('Yellow Detection', frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
