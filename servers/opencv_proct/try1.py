import cv2
import requests
import numpy as np

# Open the default camera (usually the first camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break

    # Encode the frame in JPEG format before sending
    ret, buffer = cv2.imencode('.jpg', frame)
    
    if not ret:
        print("Error: Failed to encode frame.")
        break

    # Send the frame to the server as a binary file
    response = requests.post("http://172.17.0.5:8080/post-data", files={'image': buffer.tobytes()})

    # Check if the response is successful
    if response.status_code == 200:
        # Convert response content (binary) back into a NumPy array
        npimg = np.frombuffer(response.content, np.uint8)
        print(npimg)
        # Decode the image
        processed_frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        print("received frame")
        # Display the processed frame
        cv2.imshow('Processed Camera Feed', processed_frame)
    else:
        print("Error: Failed to receive frame from server.")
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
