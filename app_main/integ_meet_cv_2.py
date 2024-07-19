import pygame
import socket
import json
import base64
import io
from PIL import Image
import time
import uuid

import socketio


import cv2
import base64
import io

def capture_video():
    # Initialize the camera
    cap = cv2.VideoCapture(0)

    # Capture a frame
    ret, frame = cap.read()

    # Encode the frame as a JPEG image
    _, buffer = cv2.imencode('.jpg', frame)

    # Convert the image to a base64-encoded string
    video_data = base64.b64encode(buffer).decode('utf-8')

    # Release the camera
    cap.release()

    return video_data
# Initialize Pygame
pygame.init()

# Set up the video conference window
window_width = 800
window_height = 600
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Video Conference")

# Connect to the Flask server
sio = socketio.Client()

@sio.event
def connect():
    print("Connected to the server")
    client_id = str(uuid.uuid4())
    sio.emit('client_connected', client_id)
    return client_id

@sio.event
def disconnect():
    print("Disconnected from the server")

@sio.event
def client_joined(data):
    print(f"Client {data['client_id']} joined the room {data['room_id']}")

@sio.event
def client_left(data):
    print(f"Client {data['client_id']} left the room {data['room_id']}")

@sio.event
def video_update(data):
    client_id = data['client_id']
    video_data = data['video_data']
    image = Image.open(io.BytesIO(base64.b64decode(video_data)))
    image = pygame.image.fromstring(image.tobytes(), image.size, image.mode)
    window.blit(image, (0, 0))
    pygame.display.flip()

def main():
    sio.connect('https://s.intellirecruit.ai')
    client_id = sio.get_sid()
    sio.emit('join_room', {'room_id': 'my_room', 'client_id': client_id})

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                sio.emit('leave_room', {'room_id': 'my_room', 'client_id': client_id})
                sio.disconnect()

        # Capture video from the user's camera and send it to the server
        # (This part is not implemented in the provided code)
        video_data = capture_video()
        sio.emit('video_data', {'room_id': 'my_room', 'client_id': client_id, 'video_data': video_data})

        time.sleep(0.1)

    pygame.quit()

if __name__ == "__main__":
    main()


# import asyncio
# import json
# import cv2
# import numpy as np
# import socketio
# import base64
# import pygame
# from threading import Lock
# import logging

# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# SERVER_URL = "https://s.intellirecruit.ai"
# CLIENT_ID = None
# ROOM_ID = "video_conference_room"

# outgoing_frame = None
# incoming_frames = {}
# frame_lock = Lock()

# sio = socketio.AsyncClient(logger=True, engineio_logger=True)

# @sio.event
# async def connect():
#     logger.info("Connected to server")

# @sio.event
# async def disconnect():
#     logger.info("Disconnected from server")

# @sio.event
# def client_connected(client_id):
#     global CLIENT_ID
#     CLIENT_ID = client_id
#     logger.info(f"Received client ID: {CLIENT_ID}")

# @sio.event
# def client_joined(data):
#     logger.info(f"Client {data['client_id']} joined room {data['room_id']}")

# @sio.event
# def client_left(data):
#     logger.info(f"Client {data['client_id']} left room {data['room_id']}")
#     with frame_lock:
#         incoming_frames.pop(data['client_id'], None)

# @sio.event
# def video_update(data):
#     if data['client_id'] != CLIENT_ID:
#         video_data = base64.b64decode(data['video_data'])
#         nparr = np.frombuffer(video_data, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         with frame_lock:
#             incoming_frames[data['client_id']] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# @sio.event
# def connect_error(data):
#     logger.error(f"Connection error: {data}")

# async def capture_and_send_video():
#     cap = cv2.VideoCapture(0)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             logger.error("Failed to capture frame")
#             break

#         _, buffer = cv2.imencode('.jpg', frame)
#         jpg_as_text = base64.b64encode(buffer).decode('utf-8')

#         with frame_lock:
#             global outgoing_frame
#             outgoing_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         try:
#             await sio.emit('video_data', {
#                 'room_id': ROOM_ID,
#                 'client_id': CLIENT_ID,
#                 'video_data': jpg_as_text
#             })
#         except Exception as e:
#             logger.error(f"Error sending video data: {e}")

#         await asyncio.sleep(1/30)  # Limit to 30 FPS

#     cap.release()

# def display_video_pygame():
#     pygame.init()
#     screen = pygame.display.set_mode((1280, 720))
#     pygame.display.set_caption("Video Conference")
#     clock = pygame.time.Clock()

#     running = True
#     while running:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False

#         screen.fill((0, 0, 0))  # Clear the screen

#         with frame_lock:
#             if outgoing_frame is not None:
#                 outgoing_surface = pygame.surfarray.make_surface(np.rot90(outgoing_frame))
#                 screen.blit(pygame.transform.scale(outgoing_surface, (640, 360)), (0, 0))

#             for i, (client_id, frame) in enumerate(incoming_frames.items()):
#                 surface = pygame.surfarray.make_surface(np.rot90(frame))
#                 screen.blit(pygame.transform.scale(surface, (640, 360)), (640 if i % 2 == 1 else 0, 360 if i // 2 == 1 else 0))

#         pygame.display.flip()
#         clock.tick(30)  # Limit to 30 FPS

#     pygame.quit()

# async def main():
#     try:
#         logger.info(f"Attempting to connect to server at {SERVER_URL}")
#         await sio.connect(SERVER_URL, wait_timeout=10)
#         logger.info("Connected successfully")
        
#         # Wait for the client ID to be set
#         timeout = 10
#         while CLIENT_ID is None and timeout > 0:
#             await asyncio.sleep(0.1)
#             timeout -= 0.1
        
#         if CLIENT_ID is None:
#             logger.error("Timed out waiting for client ID")
#             return

#         logger.info(f"Joining room {ROOM_ID} with client ID: {CLIENT_ID}")
#         await sio.emit('join_room', {'room_id': ROOM_ID, 'client_id': CLIENT_ID})

#         video_task = asyncio.create_task(capture_and_send_video())
#         display_task = asyncio.to_thread(display_video_pygame)

#         await asyncio.gather(video_task, display_task)

#     except socketio.exceptions.ConnectionError as e:
#         logger.error(f"Failed to connect to server: {e}")
#     except Exception as e:
#         logger.error(f"An error occurred: {e}")
#     finally:
#         if sio.connected:
#             await sio.disconnect()
#         logger.info("Disconnected from server")

# if __name__ == "__main__":
#     asyncio.run(main())