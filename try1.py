import cv2
import numpy as np
import dlib
import math
import mediapipe as mp
from collections import deque
from pymongo import MongoClient
from datetime import datetime
import tensorflow as tf

print('Starting')
#############################################


import cv2
import numpy as np
from tf.keras.models import load_model
from tf.keras.preprocessing.image import img_to_array

# Load pre-trained parameters for the cascade classifier and the emotion detection model
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
classifier = load_model('app_main\model.h5')  # Replace 'model.h5' with the path to your model file
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']  # Emotion labels

def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y - 10)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    return frame



fl = (615.7648315185546, 615.6675785709508)
pp = (320.3119237464785, 242.33699852535715)
dc = np.array([[-0.0013841792342825923, 0.0005103200531089663, -0.007144587675239075, 0.0026517804999418816, -0.0035477400857502327]]) 


print("1")
# # Connect to the MongoDB server
client = MongoClient("mongodb://localhost:27017/")
db = client['proctor']
mycollection = db['data1']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# print("Connecting to MongoDB")
print("2")
def db_append(sensor_data):
    document = {
        'log': sensor_data,
        'timestamp': datetime.utcnow()
    }
    mycollection.insert_one(document)

print("3")
# # Known width and distance parameters
KNOWN_WIDTH = 11.0
KNOWN_DISTANCE = 10.0
FRAMES_FOR_MOVING_AVERAGE = 800

# # Function to calculate focal length
def calculate_focal_length(pixel_width):
    print(pixel_width )
    print(KNOWN_DISTANCE)
    print(KNOWN_WIDTH)
    print("Calculating focal length....", (pixel_width * KNOWN_DISTANCE) / KNOWN_WIDTH)
    return (pixel_width * KNOWN_DISTANCE) / KNOWN_WIDTH

# # Function to calculate distance of the object from the camera
def calculate_distance(focal_length, pixel_width):
    return (KNOWN_WIDTH * focal_length) / pixel_width

# # Function to zoom the image
def zoom(image, zoom_factor):
    height, width = image.shape[:2]
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
    zoomed = cv2.resize(image, (new_width, new_height))
    top = (new_height - height) // 2
    left = (new_width - width) // 2
    zoomed = zoomed[top:top+height, left:left+width]
    return zoomed

# # Function to detect the object in the image
def detect_object(image):
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print("????????????????",hsv, lower_red, upper_red,mask, contours)
    if contours:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return w
    else:
        print("PPPPPPPPP")
        return None
    
print("4")
def detect_multiple_faces(fr, zoomed_frame):
    print("t")
    print(fr)
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    print("t2")
    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(fr, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Check the number of faces detected
    num_faces = len(faces)
    if num_faces > 1:
        cv2.putText(zoomed_frame, "Alert: More than one face detected!", (100, 850), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2)
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

print("5")

# print("loading model...")
# # Load liveness detector model
model = tf.keras.models.load_model(r"app_main\best_model_16_11pm.h5")
print("model loaded")
# # Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
print("6..................")
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
print("7")
mp_drawing = mp.solutions.drawing_utils
print("8")
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 0, 0))
print("9")
# # print("python")
# # Load the pre-trained face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"app_main\shape_predictor_68_face_landmarks.dat")
print("10")
# # Function to calculate the head pose
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

# print("2")
# # Capture video from webcam
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# print("11")
# Initialize moving average filters
left_filter = MovingAverageFilter(window_size=3)
right_filter = MovingAverageFilter(window_size=3)
# print("12")
# Previous x-coordinate of left and right iris
prev_left_x = None
prev_left_y = None
prev_right_x = None
prev_right_y = None
# prev_right_y = None
# prev_left_y = None

# Initialize mouth status variables
mouth_open = False
mouth_counter = 0

# # Assume we have the pixel width from the first frame
# ret, frame = cap.read()
# pixel_width = detect_object(frame)

# # Calculate the focal length
# focal_length = calculate_focal_length(pixel_width)

# # Initialize deque for storing recent distances
recent_distances = deque(maxlen=FRAMES_FOR_MOVING_AVERAGE)
# print("13")

def estimate_distance(landmarks):
    # Calculate the distance between the eye corners
    left_eye_corner = landmarks.part(36).x, landmarks.part(36).y
    right_eye_corner = landmarks.part(39).x, landmarks.part(39).y
    eye_distance = np.linalg.norm(np.array(left_eye_corner) - np.array(right_eye_corner))

    # Assuming the average eye distance is 6.3 cm at 1 meter
    avg_eye_distance = 6.3
    distance = avg_eye_distance / eye_distance

    return distance

def calculate_actual_distance(rgb_frame, frame):
     # Detect faces in the frame
    faces = face_cascade.detectMultiScale(rgb_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Detect facial landmarks
        face_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        landmarks = predictor(rgb_frame, face_rect)

        # Estimate the distance between the face and the camera
        distance = estimate_distance(landmarks)
        print(f"Distance: {distance:.2f} meters")
        cv2.putText(frame, f"Actual Distance: {distance:.2f} meters", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        
        if distance > 0.13:
            cv2.putText(frame, f"please stay close to the screen and maintain stability", (50, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            db_append("please stay close to the screen and maintain stability")
            
        else:
            pass  
   
   
# lx=[]
# ly=[]
# rx=[]
# ry=[]

def process_frame(incoming_frame):
    print("Processing frame")
    
    # if len(lx) is 2:
    #     lx.pop(0)
    # if len(ly) is 2:
    #     ly.pop(0)
    # if len(rx) is 2:
    #     rx.pop(0)
    # if len(ry) is 2:
    #     ry.pop(0)
    global prev_left_x, prev_left_y, prev_right_x, prev_right_y
    pixel_width = detect_object(incoming_frame)
    print("1.1")
    print("pixel_width", pixel_width)
    # import time
    # time.sleep(10)
    
# Calculate the focal length
    focal_length = calculate_focal_length(pixel_width)
    # return incoming_frame
    
    print("1.2")
    frame = incoming_frame
    height, width, _ = frame.shape
    zoom_factor = 1
    start_x = max(0, int(width / 2 - (width / zoom_factor / 2)))
    start_y = max(0, int(height / 2 - (height / zoom_factor / 2)))
    end_x = min(width, int(width / 2 + (width / zoom_factor / 2)))
    end_y = min(height, int(height / 2 + (height / zoom_factor / 2)))
    zoomed_frame = frame[start_y:end_y, start_x:end_x]
    rgb_frame = cv2.cvtColor(zoomed_frame, cv2.COLOR_BGR2RGB)
    print("1.3")
    
    detect_multiple_faces(rgb_frame, zoomed_frame)
    print("1.4")
    calculate_actual_distance(rgb_frame,frame)
    print("1.5")
    
    # # Detect the object and get its pixel width
    pixel_width = detect_object(frame)
    print("1.6")
    
    if pixel_width is not None:
        distance = calculate_distance(focal_length, pixel_width)
        zoom_factor = 1.0 + distance * 0.001
        recent_distances.append(distance)
        avg_distance = sum(recent_distances) / len(recent_distances)
        zoomed_frame = zoom(frame, zoom_factor)
        # if distance > 10 and avg_distance > 13:
            # cv2.putText(zoomed_frame, "please stay close to the screen and maintain stability", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            # db_append("please stay close to the screen and maintain stability")
        cv2.putText(zoomed_frame, f"Avg Distance: {avg_distance:.2f} inches", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 100), 2)
        cv2.putText(zoomed_frame, f"Distance: {distance:.2f} inches", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2)
    
    

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
            print("reached")
            # if lip_distance > 15:
            #     mouth_counter += 1
            #     if mouth_counter > 5:
            #         mouth_open = True
            # else:
            #     mouth_open = False
            #     mouth_counter = 0
            
            
            
            mouth_open = False
            mouth_counter = 0
            MOUTH_COUNTER_THRESHOLD = 0


            if lip_distance > 15:
                mouth_counter += 1
                if mouth_counter > MOUTH_COUNTER_THRESHOLD:
                    mouth_open = True
            else:
                mouth_counter -= 1
                if mouth_counter <= 0:
                    mouth_open = False
                    mouth_counter = 0
            
            
            if mouth_open:
                cv2.putText(zoomed_frame, "Mouth closed", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                db_append("Mouth closed")
            else:
                cv2.putText(zoomed_frame, "Mouth open", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 200), 2)
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
            
            # if lx is not []:

            #     prev_left_x = lx[0]
            #     prev_left_y = ly[0]
            #     prev_right_x = rx[0]
            #     prev_right_y = ry[0]
                
            # else:
            #     pass
            
            # print("/////////////////////////",prev_left_x, prev_right_y, prev_left_y, prev_right_x)
            
            if prev_left_x is not None and prev_left_y is not None and prev_right_x is not None and prev_right_y is not None:
            #     print("hello")
                delta_left_x = curr_left_x - prev_left_x
                delta_left_y = curr_left_y - prev_left_y
                delta_right_x = curr_right_x - prev_right_x
                delta_right_y = curr_right_y - prev_right_y
                # print(,delta_left_x, delta_right_x, delta_right_y, delta_left_y)
                if delta_left_x > 3:
                    cv2.putText(zoomed_frame, "Left iris moved right", (800, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
                    print("Right iris moved")
                    db_append("Left iris moved right")
                elif delta_left_x < -3:
                    cv2.putText(zoomed_frame, "Left iris moved left", (800, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
                    db_append("Left iris moved left")

                if delta_left_y > 3:
                    cv2.putText(zoomed_frame, "Left iris moved down", (800, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 1)
                    db_append("Left iris moved down")
                elif delta_left_y < -3:
                    cv2.putText(zoomed_frame, "Left iris moved up", (800, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 1)
                    db_append("Left iris moved up")

                if delta_right_x > 3:
                    cv2.putText(zoomed_frame, "Right iris moved right", (800, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 0), 1)
                    db_append("Right iris moved right")
                elif delta_right_x < -3:
                    cv2.putText(zoomed_frame, "Right iris moved left", (800, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 0), 1)
                    db_append("Right iris moved left")

                if delta_right_y > 3:
                    cv2.putText(zoomed_frame, "Right iris moved down", (800, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
                    db_append("Right iris moved down")
                elif delta_right_y < -3:
                    cv2.putText(zoomed_frame, "Right iris moved up", (800, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
                    db_append("Right iris moved up")

            prev_left_x = curr_left_x
            prev_left_y = curr_left_y
            prev_right_x = curr_right_x
            prev_right_y = curr_right_y
            
            # lx.append(prev_left_x)
            # rx.append(prev_right_x)
            # ly.append(prev_left_y)
            # ry.append(prev_right_y)

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
            color = (0, 165, 255) if label == "Real" else (0, 0, 255)
            cv2.putText(zoomed_frame, f"Liveness: {label}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            db_append(f"Liveness: {label}")



    print("5")
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
        print("6")
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
        
        print(direction)
        cv2.putText(zoomed_frame, "Face_direction:"+ direction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (19, 69, 139), 2)
        print("6.5")
        zoomed_frame= detect_emotion(zoomed_frame)
    print("Zoomed frame: ", zoomed_frame)
    return zoomed_frame
    # return zoomed_frame, prev_right_y, prev_left_y, prev_right_x, prev_left_x
    # print(zoomed_frame)
                # Display the resulting frame
#     cv2.imshow('Integrated Detection', zoomed_frame)
# #     print("8")
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # # Release the capture
# cap.release()
# cv2.destroyAllWindows()