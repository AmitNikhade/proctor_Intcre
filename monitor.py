import cv2
import numpy as np
import dlib
import math
import mediapipe as mp
from collections import deque
from pymongo import MongoClient
from datetime import datetime
import tensorflow as tf

# Connect to the MongoDB server
client = MongoClient("mongodb://localhost:27017/")
db = client['proctor']
mycollection = db['data1']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def db_append(sensor_data):
    document = {
        'log': sensor_data,
        'timestamp': datetime.utcnow()
    }
    mycollection.insert_one(document)

# Known width and distance parameters
KNOWN_WIDTH = 11.0
KNOWN_DISTANCE = 100.0
FRAMES_FOR_MOVING_AVERAGE = 800

# Function to calculate focal length
def calculate_focal_length(pixel_width):
    return (pixel_width * KNOWN_DISTANCE) / KNOWN_WIDTH

# Function to calculate distance of the object from the camera
def calculate_distance(focal_length, pixel_width):
    return (KNOWN_WIDTH * focal_length) / pixel_width

# Function to zoom the image
def zoom(image, zoom_factor):
    height, width = image.shape[:2]
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
    zoomed = cv2.resize(image, (new_width, new_height))
    top = (new_height - height) // 2
    left = (new_width - width) // 2
    zoomed = zoomed[top:top+height, left:left+width]
    return zoomed

# Function to detect the object in the image
def detect_object(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return w
    else:
        return None

def detect_multiple_faces(fr):
    faces = face_cascade.detectMultiScale(
        fr,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Check the number of faces detected
    num_faces = len(faces)
    if num_faces > 1:
        cv2.putText(zoomed_frame, "Alert: More than one face detected!", (50, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print(f"Alert: More than one face detected! ({num_faces} faces)")

class MovingAverageFilter:
    def __init__(self, window_size):
        self.window_size = window_size
        self.data = []

    def apply(self, value):
        self.data.append(value)
        if len(self.data) > self.window_size:
            self.data.pop(0)
        return sum(self.data) / len(self.data)

# Load liveness detector model
model = tf.keras.models.load_model("best_model_16_11pm.h5")

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 0, 0))

# Load the pre-trained face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to calculate the head pose
def get_head_pose(shape):
    image_pts = np.float32([shape[30], shape[8], shape[36], shape[45], shape[48], shape[54]])

    model_pts = np.float32([[0.0, 0.0, 0.0],       # Nose tip
                            [0.0, -330.0, -65.0],  # Chin
                            [-225.0, 170.0, -135.0],  # Left eye left corner
                            [225.0, 170.0, -135.0],   # Right eye right corner
                            [-150.0, -150.0, -125.0],  # Left Mouth corner
                            [150.0, -150.0, -125.0]])  # Right mouth corner

    focal_length = 500
    center = (shape[30][0], shape[30][1])

    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype="double")

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_pts, image_pts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose

    (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    p1 = (int(image_pts[0][0]), int(image_pts[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    return p1, p2, rotation_vector

# Capture video from webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Initialize moving average filters
left_filter = MovingAverageFilter(window_size=3)
right_filter = MovingAverageFilter(window_size=3)

# Previous x-coordinate of left and right iris
prev_left_x = None
prev_right_x = None

# Initialize mouth status variables
mouth_open = False
mouth_counter = 0

# Assume we have the pixel width from the first frame
ret, frame = cap.read()
pixel_width = detect_object(frame)

# Calculate the focal length
focal_length = calculate_focal_length(pixel_width)

# Initialize deque for storing recent distances
recent_distances = deque(maxlen=FRAMES_FOR_MOVING_AVERAGE)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    zoom_factor = 1
    start_x = max(0, int(width / 2 - (width / zoom_factor / 2)))
    start_y = max(0, int(height / 2 - (height / zoom_factor / 2)))
    end_x = min(width, int(width / 2 + (width / zoom_factor / 2)))
    end_y = min(height, int(height / 2 + (height / zoom_factor / 2)))
    zoomed_frame = frame[start_y:end_y, start_x:end_x]
    rgb_frame = cv2.cvtColor(zoomed_frame, cv2.COLOR_BGR2RGB)
    detect_multiple_faces(rgb_frame)

    # Detect the object and get its pixel width
    pixel_width = detect_object(frame)

    if pixel_width is not None:
        distance = calculate_distance(focal_length, pixel_width)
        zoom_factor = 1.0 + distance * 0.001
        recent_distances.append(distance)
        avg_distance = sum(recent_distances) / len(recent_distances)
        zoomed_frame = zoom(frame, zoom_factor)
        if distance > 150 and avg_distance > 250:
            cv2.putText(zoomed_frame, "please stay close to the screen and maintain stability", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            db_append("please stay close to the screen and maintain stability")
        cv2.putText(zoomed_frame, f"Avg Distance: {avg_distance:.2f} inches", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(zoomed_frame, f"Distance: {distance:.2f} inches", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(zoomed_frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION, drawing_spec, drawing_spec)

            upper_lip_landmarks = [13, 14, 15]
            lower_lip_landmarks = [16, 17, 18]
            for idx in upper_lip_landmarks:
                pos = face_landmarks.landmark[idx]
                x = int(pos.x * width)
                y = int(pos.y * height)
                cv2.circle(zoomed_frame, (x, y), 2, (225, 0, 0), -1)
            for idx in lower_lip_landmarks:
                pos = face_landmarks.landmark[idx]
                x = int(pos.x * width)
                y = int(pos.y * height)
                cv2.circle(zoomed_frame, (x, y), 2, (225, 0, 0), -1)

            upper_lip_y = int(face_landmarks.landmark[upper_lip_landmarks[1]].y * height)
            lower_lip_y = int(face_landmarks.landmark[lower_lip_landmarks[1]].y * height)
            lip_distance = lower_lip_y - upper_lip_y

            if lip_distance > 20:
                mouth_counter += 1
                if mouth_counter > 5:
                    mouth_open = True
            else:
                mouth_open = False
                mouth_counter = 0

            if mouth_open:
                cv2.putText(zoomed_frame, "Mouth closed", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                db_append("Mouth closed")
            else:
                cv2.putText(zoomed_frame, "Mouth open", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                db_append("Mouth open")

            left_iris_landmarks = [468, 469, 470, 471]
            right_iris_landmarks = [472, 473, 474, 475, 476, 477]

            for idx in left_iris_landmarks:
                pos = face_landmarks.landmark[idx]
                x = int(pos.x * width)
                y = int(pos.y * height)
                cv2.circle(zoomed_frame, (x, y), 2, (0, 255, 0), -1)
            for idx in right_iris_landmarks:
                pos = face_landmarks.landmark[idx]
                x = int(pos.x * width)
                y = int(pos.y * height)
                cv2.circle(zoomed_frame, (x, y), 2, (0, 255, 0), -1)

            curr_left_x = int(face_landmarks.landmark[left_iris_landmarks[0]].x * width)
            curr_left_y = int(face_landmarks.landmark[left_iris_landmarks[0]].y * height)
            curr_right_x = int(face_landmarks.landmark[right_iris_landmarks[0]].x * width)
            curr_right_y = int(face_landmarks.landmark[right_iris_landmarks[0]].y * height)

            curr_left_x = left_filter.apply(curr_left_x)
            curr_left_y = left_filter.apply(curr_left_y)
            curr_right_x = right_filter.apply(curr_right_x)
            curr_right_y = right_filter.apply(curr_right_y)

            if prev_left_x is not None and prev_left_y is not None and prev_right_x is not None and prev_right_y is not None:
                delta_left_x = curr_left_x - prev_left_x
                delta_left_y = curr_left_y - prev_left_y
                delta_right_x = curr_right_x - prev_right_x
                delta_right_y = curr_right_y - prev_right_y

                if delta_left_x > 3:
                    cv2.putText(zoomed_frame, "Left iris moved right", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    db_append("Left iris moved right")
                elif delta_left_x < -3:
                    cv2.putText(zoomed_frame, "Left iris moved left", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    db_append("Left iris moved left")

                if delta_left_y > 3:
                    cv2.putText(zoomed_frame, "Left iris moved down", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    db_append("Left iris moved down")
                elif delta_left_y < -3:
                    cv2.putText(zoomed_frame, "Left iris moved up", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    db_append("Left iris moved up")

                if delta_right_x > 3:
                    cv2.putText(zoomed_frame, "Right iris moved right", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    db_append("Right iris moved right")
                elif delta_right_x < -3:
                    cv2.putText(zoomed_frame, "Right iris moved left", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    db_append("Right iris moved left")

                if delta_right_y > 3:
                    cv2.putText(zoomed_frame, "Right iris moved down", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    db_append("Right iris moved down")
                elif delta_right_y < -3:
                    cv2.putText(zoomed_frame, "Right iris moved up", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    db_append("Right iris moved up")

            prev_left_x = curr_left_x
            prev_left_y = curr_left_y
            prev_right_x = curr_right_x
            prev_right_y = curr_right_y

            x_min = max(0, int(min([face_landmarks.landmark[i].x for i in range(468, 478)]) * width))
            y_min = max(0, int(min([face_landmarks.landmark[i].y for i in range(468, 478)]) * height))
            x_max = min(width, int(max([face_landmarks.landmark[i].x for i in range(468, 478)]) * width))
            y_max = min(height, int(max([face_landmarks.landmark[i].y for i in range(468, 478)]) * height))

            iris_frame = frame[y_min:y_max, x_min:x_max]
            resized_iris_frame = cv2.resize(iris_frame, (64, 64))
            resized_iris_frame = resized_iris_frame.astype("float") / 255.0
            resized_iris_frame = np.expand_dims(resized_iris_frame, axis=0)

            preds = model.predict(resized_iris_frame)[0]
            j = np.argmax(preds)
            label = "Real" if j == 0 else "Fake"
            color = (0, 255, 0) if label == "Real" else (0, 0, 255)
            cv2.putText(zoomed_frame, f"Liveness: {label}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            db_append(f"Liveness: {label}")

    # Detect faces
    faces = detector(rgb_frame)

    for face in faces:
        # Detect landmarks
        shape = predictor(rgb_frame, face)

        # Convert shape to numpy array
        shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

        # Draw landmarks
        for (x, y) in shape:
            cv2.circle(zoomed_frame, (x, y), 1, (0, 0, 255), -1)

        # Estimate head pose and draw line
        p1, p2, rotation_vector = get_head_pose(shape)
        cv2.line(zoomed_frame, p1, p2, (0, 255, 0), 2)

        # Calculate the angle of rotation around the y-axis and x-axis
        rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
        _, _, _, _, _, _, angles = cv2.decomposeProjectionMatrix(cv2.hconcat((rotation_matrix, np.zeros((3, 1)))))
        yaw = math.degrees(angles[1])  # Rotation around the y-axis
        pitch = math.degrees(angles[0])  # Rotation around the x-axis
        print(pitch)

        # Display direction
        direction = ""
        if yaw > 1000 and yaw < 3000:
            direction = "Right"
        elif yaw < -1000 and yaw > -3000:
            direction = "Left"
        elif pitch > 9000 and pitch < 9700:
            direction = "up"
        elif pitch < -9000 and pitch > -9700:
            direction = "Down"

        cv2.putText(zoomed_frame, direction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display the resulting frame
    cv2.imshow('Integrated Detection', zoomed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
