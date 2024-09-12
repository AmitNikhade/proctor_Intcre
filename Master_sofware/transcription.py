import cv2
import numpy as np

# Initialize video capture (0 for default camera)
cap = cv2.VideoCapture(0)

# Initialize background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Remove background from frame
    foreground = cv2.bitwise_and(frame, frame, mask=fg_mask)

    # Convert frame to grayscale for further processing
    grayscale_image = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)

    # Use Canny edge detection to highlight edges
    edges = cv2.Canny(blurred_image, 50, 150)

    # Find all the contours based on the edges
    all_contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through individual contours
    for contour in all_contours:
        # Approximate contour to a polygon
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # Only consider large contours by area
        area = cv2.contourArea(contour)
        if area > 1000:  # Minimum area threshold
            # Check if the approximated contour has 4 points (rectangle)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h

                # Filter rectangles (aspect ratio should deviate significantly from 1)
                if aspect_ratio > 1.2 or aspect_ratio < 0.8:
                    # It's a rectangle (not a square)
                    cv2.drawContours(foreground, [approx], -1, (0, 255, 0), 3)
                    cv2.putText(
                        foreground,
                        f"Rectangle (AR={aspect_ratio:.2f})",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
                else:
                    # It's a square
                    cv2.drawContours(foreground, [approx], -1, (255, 0, 0), 3)
                    cv2.putText(
                        foreground,
                        f"Square (AR={aspect_ratio:.2f})",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2,
                    )

    # Display the result frame-by-frame with background removed
    cv2.imshow("Foreground with Detected Rectangles", foreground)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
