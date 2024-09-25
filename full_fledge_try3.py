




import e_d_func
import asyncio
import json
import cv2
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
import aiohttp
import mss
from aiortc import RTCIceServer, RTCConfiguration
import logging
import traceback
import sys
from aiortc.contrib.signaling import object_from_string, object_to_string
import time
from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
# from webdriver_manager.chrome import ChromeDriverManager
import ctypes, threading

stop_event = threading.Event()
thread4 = None


import threading
import time
import ctypes
from pyee import AsyncIOEventEmitter
import asyncio
import requests

# Create an event emitter for managing events
emitter = AsyncIOEventEmitter()

# Event to signal thread termination
stop_event1 = threading.Event()

# def worker_thread(stop_event1):
#     """Worker thread that runs until stop_event is set."""
    # try:
    #     while not stop_event1.is_set():
#             print("Thread is running...")
#             time.sleep(1)  # Simulate work
#     except SystemExit:
#         print("Thread received SystemExit exception.")
#     finally:
#         print("Thread cleanup completed. Exiting...")

def async_raise(tid, exctype):
    """Raises an exception in the threads with id tid."""
    if not isinstance(tid, int):
        raise TypeError("tid must be an int")
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("Invalid thread id")
    elif res != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), None)
        raise SystemError("PyThreadState_SetAsyncExc failed")

def terminate_thread(thread):
    """Forcibly terminate the thread by raising SystemExit."""
    if not thread.is_alive():
        return
    async_raise(thread.ident, SystemExit)

# Error handling for pyee emitter
def on_error(exc):
    print(f"Error event caught: {exc}")

# Attach the error handler to the emitter
emitter.on("error", on_error)

ice_servers = [
    RTCIceServer(urls='stun:global.stun.twilio.com:3478'),  # Public STUN server
    RTCIceServer(urls='turn:global.turn.twilio.com:3478?transport=udp', 
                 credential='3H+fje6AEX9gzDyDYqdrpoTZ+WbqedlIWVPyeGX9K4w=', 
                 username='78142ed60b57fc91de5c5b8cb569380d60d59d748e56206ee70144b18471d19e'),  # TURN server with UDP transport
    RTCIceServer(urls='turn:global.turn.twilio.com:3478?transport=tcp', 
                 credential='3H+fje6AEX9gzDyDYqdrpoTZ+WbqedlIWVPyeGX9K4w=', 
                 username='78142ed60b57fc91de5c5b8cb569380d60d59d748e56206ee70144b18471d19e'),  # TURN server with TCP transport
    RTCIceServer(urls='turn:global.turn.twilio.com:443?transport=tcp', 
                 credential='3H+fje6AEX9gzDyDYqdrpoTZ+WbqedlIWVPyeGX9K4w=', 
                 username='78142ed60b57fc91de5c5b8cb569380d60d59d748e56206ee70144b18471d19e'),  # TURN server with TCP transport on port 443
]








def join_room():
    room_url = "https://intellicredence.daily.co/example-room1"
    # Initialize the Selenium WebDriver (Chrome in this case) with options
    chrome_options = Options()
    chrome_options.add_argument("--use-fake-ui-for-media-stream")  # Automatically allow mic/camera
    chrome_options.add_argument("--headless")  # Run Chrome in headless mode
    chrome_options.add_argument("--disable-gpu")  # Disable GPU acceleration (commonly used with headless)
    chrome_options.add_argument("--window-size=1920x1080")
    driver = webdriver.Chrome(options=chrome_options)

    # Open the room URL
    driver.get(room_url)

    # Keep the browser open indefinitely
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Manual interruption detected, closing the browser...")
        driver.quit()

# # Example usage:
# if __name__ == "__main__":
#     # Define the room URL
    

#     # Join the existing room and start the video conference using Selenium
    # join_room(room_url, user_name="interviewee")


logging.basicConfig(level=logging.INFO)

class ScreenShareTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.restart()

    def restart(self):
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[0]  # Capture the primary monitor
        self.frame_count = 0

    async def recv(self):
        self.frame_count += 1
        if self.frame_count % 30 == 0:  # Log every 30 frames
            logging.info(f"Sending frame {self.frame_count}")
        screenshot = self.sct.grab(self.monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        pts, time_base = await self.next_timestamp()
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame

async def reset_room(signaling_url):
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{signaling_url}/reset_room/test_room") as resp:
            if resp.status != 200:
                logging.error(f"Failed to reset room: {resp.status}")
                
                
async def check_stop_signal(signaling_url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{signaling_url}/get_stop_signal/test_room") as resp:
                if resp.status == 200:
                    stop_signal = await resp.json()
                    
                    return stop_signal.get('stop', False)
    except Exception as e:
        logging.error(f"Error checking stop signal: {e}")
    return False

async def ensure_signaling(pc, signaling_url, max_retries=5):
    for attempt in range(max_retries):
        try:
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)
            
            offer_dict = object_to_string(pc.localDescription)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{signaling_url}/offer",
                    json={"offer": offer_dict, "room": "test_room"}
                ) as resp:
                    if resp.status != 200:
                        raise Exception(f"Failed to send offer: {resp.status}")
            
            logging.info("Offer sent to signaling server")
            
            while True:
                if await check_stop_signal(signaling_url):
                    logging.info("Received stop signal from server")
                    
                    await reset_room(signaling_url)
                    await reset_peer_connection(pc)
                    await main()
                    # await pc.close()
                    # return "STOP"

                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{signaling_url}/get_answer/test_room") as resp:
                        if resp.status == 200:
                            answer_dict = await resp.json()
                            answer = object_from_string(json.dumps(answer_dict))
                            await pc.setRemoteDescription(answer)
                            logging.info("Answer received and set")
                            return "CONNECTED"
                        elif resp.status != 404:
                            raise Exception(f"Unexpected status when getting answer: {resp.status}")
                await asyncio.sleep(1)
        except Exception as e:
            logging.error(f"Signaling attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
        await asyncio.sleep(2 ** attempt)  # Exponential backoff

async def reset_peer_connection(pc):
    if pc:
        await pc.close()
#     ice_servers = [
#     RTCIceServer(urls='stun:global.stun.twilio.com:3478'),  # Public STUN server
#     RTCIceServer(urls='turn:global.turn.twilio.com:3478?transport=udp', 
#                  username='a6d3a4f08322251270fe2abd78b49110c776119fb85c9fac358b11e6f4e905dc', 
#                  credential='kYJ0CyOAj9QE1xyf7XzskmUk2sk6zLN9KkqQsRY5FF8='),  # TURN server with UDP transport
#     RTCIceServer(urls='turn:global.turn.twilio.com:3478?transport=tcp', 
#                  username='a6d3a4f08322251270fe2abd78b49110c776119fb85c9fac358b11e6f4e905dc', 
#                  credential='kYJ0CyOAj9QE1xyf7XzskmUk2sk6zLN9KkqQsRY5FF8='),  # TURN server with TCP transport
#     RTCIceServer(urls='turn:global.turn.twilio.com:443?transport=tcp', 
#                  username='a6d3a4f08322251270fe2abd78b49110c776119fb85c9fac358b11e6f4e905dc', 
#                  credential='kYJ0CyOAj9QE1xyf7XzskmUk2sk6zLN9KkqQsRY5FF8=')   # TURN server with TCP transport on port 443
# ]


    configuration = RTCConfiguration(iceServers=ice_servers)

    # Then use this configuration when creating the RTCPeerConnection
    new_pc = RTCPeerConnection(configuration=configuration)
    new_pc.addTrack(ScreenShareTrack())
    
    @new_pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logging.info(f"Connection state is {new_pc.connectionState}")
        if new_pc.connectionState == "connected":
            logging.info("Peer connection established!")
        elif new_pc.connectionState in ["failed", "disconnected", "closed"]:
            logging.warning(f"Connection state changed to {new_pc.connectionState}")
            print(f"thread status: {thread4.is_alive()}")
            if thread4.is_alive() and threading.current_thread() != thread4:
    # thread4.join()
                stop_event1.set()
                thread4.join()
            
            # thread4.join(timeout=2)
    return new_pc

async def heartbeat(pc, reset_func):
    while True:
        if pc.connectionState in ["failed", "disconnected", "closed"]:
            logging.info("Connection lost. Attempting to reconnect...")
            return  # Exit heartbeat, main loop will handle reconnection
        await asyncio.sleep(5)  # Check every 5 seconds

async def restart_ice(pc):
    logging.info("Restarting ICE")
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

async def main():
    signaling_url = "https://dash.intellirecruit.ai"  # Update this to your signaling server URL
    pc = None

    while True:
        try:
            import requests, time
            requests.post('https://dash.intellirecruit.ai/stop_client/test_room')
            
            time.sleep(5)
            await reset_room(signaling_url)
            pc = await reset_peer_connection(pc)
            await restart_ice(pc)
            result = await ensure_signaling(pc, signaling_url)
            if result == "STOP":
                logging.info("Stopping the client as requested by the server")
                break
            logging.info("Screen sharing started")
            
            while True:
                if await check_stop_signal(signaling_url):
                    
                    logging.info("Stopping the client as requested by the server")
                    await reset_room(signaling_url)
                    # if pc:
                    #     await pc.close()
                    # pc = await reset_peer_connection(None)
                    await reset_peer_connection(None)
                    await pc.close()
                    return
                await asyncio.sleep(5)  # Check every 5 seconds
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            logging.error(traceback.format_exc())
            await asyncio.sleep(5)  # Wait before retrying
            
# if __name__ == "__main__":
#     try:
#         asyncio.run(main())
#     except Exception as e:
#         print("An error occurred:::::::::::::::::::::::::::::", e)
        
def screen_s(stop_event1):
    try:
        while not stop_event1.is_set():
    # try:
            asyncio.run(main())
            time.sleep(2)
    except Exception as e:
        print("An error occurred:::::::::::::::::::::::::::::", e)
        
import subprocess
# import socketio
# import pyautogui
# import base64
import time
import io
# from PIL import Image
# import pyaudio

# sio = socketio.Client(logger=True, engineio_logger=True)
# sio1 = socketio.Client(logger=True, engineio_logger=True)



    
    
    

import sys
import io
from contextlib import redirect_stdout, redirect_stderr
# import argparse
import io
# import speech_recognition as sr
# import torch
import asyncio
import json
import cv2
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaRecorder
import websockets
from av import VideoFrame
import logging
# from datetime import datetime
# from queue import Queue
# from tempfile import NamedTemporaryFile
from sys import platform
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QLabel, QTextEdit, QDialog, QGraphicsBlurEffect, QCheckBox
from PyQt5.QtCore import Qt, QTimer, pyqtSlot,  pyqtSignal, QObject, QRunnable, QThreadPool
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPainter
from PyQt5.Qsci import QsciScintilla, QsciLexerPython
from PyQt5.QtWidgets import QGraphicsBlurEffect, QGraphicsDropShadowEffect
import threading
import elevate
import sys
import numpy as np
import traceback
import time
import e_d_func
# import try1
from threading import Lock
# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

SERVER_URL = "wss://dashboard.intellirecruit.ai/websocket"
CLIENT_ID = "client1"

frame_lock = Lock()

def custom_excepthook(exc_type, exc_value, exc_traceback):
    # Log the exception
    print("Uncaught exception", exc_type, exc_value)
    traceback.print_tb(exc_traceback)
    
    # Call the default excepthook
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = custom_excepthook

class CameraManager:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.lock = threading.Lock()

    def read_frame(self):
        with self.lock:
            ret, frame = self.cap.read()
            
            ret, buffer = cv2.imencode('.jpg', frame)
            response = requests.post("http://127.0.0.1:5000/post-data", files={'image': buffer.tobytes()})

            # Check if the response is successful
            if response.status_code == 200:
                # Convert response content (binary) back into a NumPy array
                npimg = np.frombuffer(response.content, np.uint8)
                # Decode the image
                processed_frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR) # try1.process_frame(response.dimensions)
            # print(response)
        if ret:
            return processed_frame
        return None


    def release(self):
        self.cap.release()

camera_manager = CameraManager()

class VideoTransformTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.camera_manager = camera_manager
      

    async def recv(self):
        global outgoing_frame
        pts, time_base = await self.next_timestamp()
        frame = self.camera_manager.read_frame()
        if frame is None:
            raise RuntimeError("Failed to capture frame")
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base

        return video_frame

class WebRTCClient(QObject):
    incoming_frame = pyqtSignal(np.ndarray)
    outgoing_frame = pyqtSignal(np.ndarray)

    def __init__(self, running_event):
        super().__init__()
        self.pc = None
        self.video_track = None
        self.running_event = running_event

    async def run_offer(self, pc):
        await pc.setLocalDescription(await pc.createOffer())
        logger.debug(f"Local Description: {pc.localDescription.sdp}")
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    
    async def run_answer(self, pc, offer):
        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer["sdp"], type=offer["type"]))
        print(f"Remote Description set with SDP: {offer['sdp']}")
        await pc.setLocalDescription(await pc.createAnswer())
        print(f"Local Description (Answer): {pc.localDescription.sdp}")
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


    async def consume_signaling(self, pc, websocket):
        async for message in websocket:
            print(f"Received message: {message}")
            msg = json.loads(message)
            if msg["type"] == "offer":
                answer = await self.run_answer(pc, msg["offer"])
                await websocket.send(json.dumps({"type": "answer", "answer": answer}))
                print("Sent answer")
            elif msg["type"] == "answer":
                await pc.setRemoteDescription(RTCSessionDescription(sdp=msg["answer"]["sdp"], type=msg["answer"]["type"]))
                print("Set remote description")
            elif msg["type"] == "stop":
                print("Received stop signal")
                await pc.close()
                await websocket.close()
                print("Closed WebRTC and WebSocket connections")
                await self.main_webrtc(self.running_event)
            
                break
            
    async def main_webrtc(self, running_event):
        # while True:
            recorder = MediaRecorder("received_video_client1.mp4")
            

            ice_servers = [
    RTCIceServer(urls='stun:global.stun.twilio.com:3478'),  # Public STUN server
    RTCIceServer(urls='turn:global.turn.twilio.com:3478?transport=udp', 
                 credential='3H+fje6AEX9gzDyDYqdrpoTZ+WbqedlIWVPyeGX9K4w=', 
                 username='78142ed60b57fc91de5c5b8cb569380d60d59d748e56206ee70144b18471d19e'),  # TURN server with UDP transport
    RTCIceServer(urls='turn:global.turn.twilio.com:3478?transport=tcp', 
                 credential='3H+fje6AEX9gzDyDYqdrpoTZ+WbqedlIWVPyeGX9K4w=', 
                 username='78142ed60b57fc91de5c5b8cb569380d60d59d748e56206ee70144b18471d19e'),  # TURN server with TCP transport
    RTCIceServer(urls='turn:global.turn.twilio.com:443?transport=tcp', 
                 credential='3H+fje6AEX9gzDyDYqdrpoTZ+WbqedlIWVPyeGX9K4w=', 
                 username='78142ed60b57fc91de5c5b8cb569380d60d59d748e56206ee70144b18471d19e'),  # TURN server with TCP transport on port 443
]






            configuration = RTCConfiguration(iceServers=ice_servers)

    # # Then use this configuration when creating the RTCPeerConnection
    # new_pc = RTCPeerConnection(configuration=configuration)
            self.pc = RTCPeerConnection(configuration=configuration)
            self.video_track = VideoTransformTrack()
            self.pc.addTrack(self.video_track)

            @self.pc.on("track")
            def on_track(track):
                logger.info("Receiving video track")
                if track.kind == "video":
                    recorder.addTrack(track)

                    async def display_incoming_video():
                        while running_event.is_set():
                            frame = await track.recv()
                            img = frame.to_ndarray(format="bgr24")
                            self.incoming_frame.emit(img)

                    asyncio.ensure_future(display_incoming_video())
                    logger.debug("Added video track to recorder and displaying incoming video")
            
            @self.pc.on("connectionstatechange")
            async def on_connectionstatechange():
                print(f"Connection state changed to: {self.pc.connectionState}")
                if self.pc.connectionState == "failed":
                    await self.pc.close()

            display_thread = None
        
            try:
                logger.info(f"Attempting to connect to wss://s.intellirecruit.ai/websocket")
                async with websockets.connect("wss://s.intellirecruit.ai/websocket") as websocket:
                    logger.info("Connected to WebSocket server")

                    await websocket.send(json.dumps({"type": "join", "client_id": CLIENT_ID}))
                    logger.info(f"Sent join message for {CLIENT_ID}")
                    
                    if CLIENT_ID == "client1":
                        offer = await self.run_offer(self.pc)
                        await websocket.send(json.dumps({"type": "offer", "offer": offer}))
                        logger.info("Sent offer")

                    signaling_task = asyncio.create_task(self.consume_signaling(self.pc, websocket))
                    await recorder.start()
                    print("Recorder started")

                    
                    # display_thread = asyncio.to_thread(display_video_pygame)
                
                    # Wait for the signaling task and display thread to complete
                    await asyncio.gather(signaling_task)

                    while running_event.is_set():
                        await asyncio.sleep(1)

            except websockets.exceptions.ConnectionClosed as e:
                logger.error(f"WebSocket connection closed unexpectedly: {e}")
            except Exception as e:
                logger.error(f"An error occurred: {e}")
            finally:
                logger.info("Cleaning up...")
                await recorder.stop()
                await self.pc.close()

class WebRTCWorker(QRunnable):
    def __init__(self, webrtc_client):
        super().__init__()
        self.webrtc_client = webrtc_client
        # self.running_event = running_event
        self.loop = None

    def run(self):
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.webrtc_client.main_webrtc(self.webrtc_client.running_event))
        except Exception as e:
            error_msg = f"An error occurred in the WebRTC thread: {e}\n{traceback.format_exc()}"
            logger.error(error_msg)
        finally:
            if self.loop:
                self.loop.close()
            
class VADWorker(QRunnable):
    def __init__(self, running_event):
        super().__init__()
        self.running_event = running_event

    def run(self):
        try:
            main_vad(self.running_event)
        except Exception as e:
            error_msg = f"An error occurred in the VAD thread: {e}\n{traceback.format_exc()}"
            logger.error(error_msg)
            
from PyQt5.QtCore import QThread, pyqtSignal, QPoint
import subprocess
from PyQt5.Qsci import QsciLexerPython, QsciLexerCPP, QsciLexerJava, QsciLexerJavaScript, QsciLexerHTML, QsciLexerCSS
from PyQt5.QtWidgets import QComboBox

class PingWorker(QThread):
    ping_result = pyqtSignal(str, str)  # Signal to send the ping result back to the UI thread

    def run(self):
        try:
            response = subprocess.run(
                ['ping', '-c', '1', '8.8.8.8'],  # Ping Google's DNS server
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            if response.returncode == 0:
                output = response.stdout.decode()
                latency = output.split('time=')[-1].split(' ms')[0]
                self.ping_result.emit(latency, "green")
            else:
                self.ping_result.emit("Unreachable", "red")
        except Exception as e:
            self.ping_result.emit("Error", "red")
            logger.error(f"Error checking ping: {e}")


class BlurredBackground(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(0, 0, 0, 120))  # Black with 47% opacity (120/255)
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.rect(), 15, 15)


class BlurredBackground(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(0, 0, 0, 120))  # Black with 47% opacity (120/255)
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.rect(), 15, 15)

class DisclaimerDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Disclaimer")
        self.setMinimumSize(450, 300)

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowSystemMenuHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # Create a blurred background
        self.blurred_bg = BlurredBackground(self)
        blur_effect = QGraphicsBlurEffect()
        blur_effect.setBlurRadius(15)
        self.blurred_bg.setGraphicsEffect(blur_effect)

        # Create a layout for the content
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)

        # Add a subheader label
        subheader_label = QLabel("Please read and accept the terms to proceed")
        subheader_label.setFont(QFont("Arial", 14, QFont.Bold))
        subheader_label.setAlignment(Qt.AlignCenter)
        subheader_label.setWordWrap(True)
        subheader_label.setStyleSheet("color: white;")
        layout.addWidget(subheader_label)

        # Add the disclaimer text
        disclaimer_label = QLabel("By clicking 'Agree', you confirm that you have read and agree to the terms and conditions of this application. This includes, By using this proctoring app, you consent to being recorded and monitored during your exam. The app may collect video, audio, screen activity, and user data, which will be securely stored and used solely to ensure exam integrity. It is your responsibility to have a functioning device with a webcam, microphone, and stable internet connection, and to take the exam in a quiet, private environment free from unauthorized materials or assistance. Any form of cheating will result in disciplinary action. The app provider is not liable for technical issues or privacy breaches, except in cases of gross negligence. By proceeding, you agree to these terms.")
        disclaimer_label.setWordWrap(True)
        disclaimer_label.setAlignment(Qt.AlignCenter)
        disclaimer_label.setStyleSheet("color: white;")
        layout.addWidget(disclaimer_label)

        # Add spacing
        layout.addSpacing(20)

        # Add checkbox
        self.checkbox = QCheckBox("I have read and agree to the terms and conditions")
        self.checkbox.setStyleSheet("""
            QCheckBox {
                color: white;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:unchecked {
                border: 2px solid white;
                background: transparent;
            }
            QCheckBox::indicator:checked {
                border: 2px solid #3498db;
                background: #3498db;
            }
        """)
        self.checkbox.stateChanged.connect(self.toggle_button)
        layout.addWidget(self.checkbox)

        # Add spacing
        layout.addSpacing(10)

        # Add the agree button
        self.agree_button = QPushButton("Agree")
        self.agree_button.setFixedSize(100, 40)
        self.agree_button.clicked.connect(self.accept)
        self.agree_button.setEnabled(False)  # Initially disabled
        self.agree_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
        """)

        # Center the agree button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.agree_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)

    def toggle_button(self, state):
        self.agree_button.setEnabled(state == Qt.Checked)

    def resizeEvent(self, event):
        self.blurred_bg.setGeometry(self.rect())
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.oldPos = event.globalPos()

    def mouseMoveEvent(self, event):
        if hasattr(self, 'oldPos'):
            delta = QPoint(event.globalPos() - self.oldPos)
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.oldPos = event.globalPos()
        
class FullScreenWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.running_event = threading.Event()
        self.running_event.set()
        
        # self.setWindowTitle("Kiosk App")
        # self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.WindowDoesNotAcceptFocus)
        self.setFixedSize(1920, 1080)
        
        self.setWindowTitle("Kiosk App")
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                border: 1px solid #555;
            }
        """)
        
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left side layout
        self.left_layout = QVBoxLayout()
        
        self.code_editor = QsciScintilla(self)
        self.setup_editor()
        self.left_layout.addWidget(self.code_editor)
        
        self.output_display = QTextEdit(self)
        self.output_display.setReadOnly(True)
        self.output_display.setStyleSheet("""
            QTextEdit {
                background-color: #2e2e2e;
                color: #ffffff;
                border: 1px solid #555;
            }
        """)
        self.output_display.setFixedHeight(200)
        self.left_layout.addWidget(self.output_display)
        
        self.run_layout = QHBoxLayout()
        self.run_button = QPushButton("Run Code", self)
        self.run_button.setFixedSize(100, 40)
        self.run_button.clicked.connect(self.run_code)
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #007BFF;
                color: #ffffff;
                border: 1px solid #0056b3;
                padding: 5px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #003f7f;
            }
        """)
        self.left_layout.addWidget(self.run_button)
        
        main_layout.addLayout(self.left_layout)
        
        # Right side layout
        right_layout = QVBoxLayout()
        
        video_and_log_layout = QHBoxLayout()
        video_layout = QVBoxLayout()
        
        self.outgoing_camera_label = QLabel(self)
        self.outgoing_camera_label.setStyleSheet("background-color: #2e2e2e; border: 1px solid #555;")
        self.outgoing_camera_label.setFixedSize(400, 250)
        video_layout.addWidget(self.outgoing_camera_label)

        self.incoming_camera_label = QLabel(self)
        self.incoming_camera_label.setStyleSheet("background-color: #2e2e2e; border: 1px solid #555;")
        self.incoming_camera_label.setFixedSize(400, 250)
        video_layout.addWidget(self.incoming_camera_label)

        video_and_log_layout.addLayout(video_layout)

        self.log_console = QTextEdit(self)
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #555;
                font-family: Consolas, monospace;
                font-size: 12px;
            }
        """)
        self.log_console.setFixedSize(400, 250)
        video_layout.addWidget(self.log_console)
        
        right_layout.addLayout(video_and_log_layout)

        self.warning_label = QLabel(self)
        self.warning_label.setStyleSheet("color: #ff6b6b; font-size: 16px;")
        self.warning_label.setFixedHeight(30)
        right_layout.addWidget(self.warning_label)
        
        main_layout.addLayout(right_layout)

        # Logo
        self.logo_label = QLabel(self)
        logo_pixmap = QPixmap("sony.jpeg")
        self.logo_label.setPixmap(logo_pixmap.scaled(100, 50, Qt.KeepAspectRatio))
        self.logo_label.setFixedSize(100, 50)
        right_layout.addWidget(self.logo_label)

        # Ping label
        self.ping_label = QLabel(self)
        self.ping_label.setStyleSheet("color: #00ff00; font-size: 16px;")
        self.ping_label.setText("Ping: Calculating...")
        self.ping_label.setAlignment(Qt.AlignLeft)
        right_layout.addWidget(self.ping_label)

        # Timer label
        self.timer_label = QLabel(self)
        self.timer_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 24px;
                font-weight: bold;
                background-color: #333333;
                border-radius: 10px;
                padding: 10px;
                qproperty-alignment: AlignCenter;
                box-shadow: 3px 3px 15px rgba(0, 0, 0, 0.5);
            }
        """)
        self.timer_label.setFixedSize(200, 60)
        self.left_layout.addWidget(self.timer_label)

        # Initialize components
        self.init_components()

    def init_components(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_outgoing_frame)
        self.timer.start(100)

        self.webrtc_client = WebRTCClient(self.running_event)
        self.webrtc_client.incoming_frame.connect(self.update_incoming_frame)
        
        self.thread_pool = QThreadPool()
        self.webrtc_worker = WebRTCWorker(self.webrtc_client)
        self.thread_pool.start(self.webrtc_worker)

        self.vad_worker = VADWorker(self.running_event)
        self.thread_pool.start(self.vad_worker)

        self.ping_worker = PingWorker()
        self.ping_worker.ping_result.connect(self.update_ping)
        self.ping_worker.start()

        self.timer_ping = QTimer(self)
        self.timer_ping.timeout.connect(self.start_ping_worker)
        self.timer_ping.start(5000)

        self.elapsed_time = 0
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self.update_timer)
        self.ui_timer.start(1000)

    def setup_editor(self):
        self.code_editor.setUtf8(True)
        self.code_editor.setFont(QFont("Consolas", 12))
        self.code_editor.setMarginsFont(QFont("Consolas", 12))
        self.code_editor.setMarginLineNumbers(1, True)
        self.code_editor.setMarginWidth(1, 50)
        self.code_editor.setBraceMatching(QsciScintilla.SloppyBraceMatch)
        self.code_editor.setCaretLineVisible(True)
        
        self.code_editor.setPaper(QColor("#1e1e1e"))
        self.code_editor.setColor(QColor("#ffffff"))
        self.code_editor.setMarginsBackgroundColor(QColor("#2e2e2e"))
        self.code_editor.setMarginsForegroundColor(QColor("#888888"))
        self.code_editor.setCaretForegroundColor(QColor("#ffffff"))
        self.code_editor.setCaretLineBackgroundColor(QColor("#2a2a2a"))
        
        # Create lexers for different languages
        self.lexers = {
            "Python": QsciLexerPython(),
            "C++": QsciLexerCPP(),
            "Java": QsciLexerJava(),
            "JavaScript": QsciLexerJavaScript(),
            "HTML": QsciLexerHTML(),
            "CSS": QsciLexerCSS()
        }
        
        # Set up each lexer with custom colors
        for lexer in self.lexers.values():
            lexer.setDefaultPaper(QColor("#1e1e1e"))
            lexer.setDefaultColor(QColor("#f8f8f2"))
            lexer.setColor(QColor("#66d9ef"), QsciLexerPython.Keyword)
            lexer.setColor(QColor("#75715e"), QsciLexerPython.Comment)
            lexer.setColor(QColor("#e6db74"), QsciLexerPython.DoubleQuotedString)
            lexer.setColor(QColor("#e6db74"), QsciLexerPython.SingleQuotedString)
            lexer.setColor(QColor("#ae81ff"), QsciLexerPython.Number)
            lexer.setColor(QColor("#a6e22e"), QsciLexerPython.FunctionMethodName)
            lexer.setColor(QColor("#f92672"), QsciLexerPython.Operator)
            lexer.setColor(QColor("#fd971f"), QsciLexerPython.Identifier)
        
        # Set initial lexer to Python
        self.set_lexer("Python")
        
        # Create a language selection dropdown
        self.language_selector = QComboBox(self)
        self.language_selector.addItems(self.lexers.keys())
        self.language_selector.currentTextChanged.connect(self.set_lexer)
        self.language_selector.setFixedSize(200, 60)
        # Add the language selector to the layout
        self.language_selector.setStyleSheet("""
        QComboBox {
            background-color: #2e2e2e;
            color: #ffffff;
            border: 1px solid #555;
            padding: 5px;
            border-radius: 3px;
        }
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 25px;
            border-left-width: 1px;
            border-left-color: #555;
            border-left-style: solid;
        }
        QComboBox::down-arrow {
            image: url(down_arrow.png);  # You'll need to provide this image
        }
        QComboBox QAbstractItemView {
            background-color: #2e2e2e;
            color: #ffffff;
            selection-background-color: #3e3e3e;
        }
    """)
    
        self.left_layout.addWidget(self.language_selector)

    def set_lexer(self, language):
        self.code_editor.setLexer(self.lexers[language])

    def run_code(self):
        code = self.code_editor.text()
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        try:
            local_vars = {}
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(code, globals(), local_vars)
            
            output = stdout_buffer.getvalue()
            error_output = stderr_buffer.getvalue()
            
            if output:
                self.output_display.setText(f"Output:\n{output}")
            elif error_output:
                self.output_display.setText(f"Errors:\n{error_output}")
            elif local_vars:
                var_output = "\n".join(f"{k} = {v}" for k, v in local_vars.items() if not k.startswith("__"))
                self.output_display.setText(f"No print output. Variables created/modified:\n{var_output}")
            else:
                self.output_display.setText("Code executed successfully (no output or variables modified).")
            
        except Exception as e:
            error_msg = f"Error executing code: {str(e)}\n\n{traceback.format_exc()}"
            self.output_display.setText(error_msg)
            logger.error(error_msg)
        finally:
            stdout_buffer.close()
            stderr_buffer.close()

    @pyqtSlot()
    def update_outgoing_frame(self):
        try:
            frame = camera_manager.read_frame()
            if frame is not None:
                image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                self.outgoing_camera_label.setPixmap(QPixmap.fromImage(image))
                self.webrtc_client.outgoing_frame.emit(frame)
            else:
                logger.warning("Failed to read frame from camera")
        except Exception as e:
            logger.error(f"Error in update_outgoing_frame: {e}")

    @pyqtSlot(np.ndarray)
    def update_incoming_frame(self, frame):
        image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        self.incoming_camera_label.setPixmap(QPixmap.fromImage(image))

    def start_ping_worker(self):
        if not self.ping_worker.isRunning():
            self.ping_worker.start()

    def update_ping(self, latency, color):
        self.ping_label.setText(f"Ping: {latency} ms")
        self.ping_label.setStyleSheet(f"color: {color}; font-size: 16px;")

    def update_timer(self):
        self.elapsed_time += 1
        hours, remainder = divmod(self.elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.timer_label.setText(f"Time: {hours:02}:{minutes:02}:{seconds:02}")

    def closeEvent(self, event):
        logger.info("Closing application...")
        # e_d_func.enable()
        self.running_event.clear()
        self.timer.stop()
        self.thread_pool.waitForDone(5000)
        camera_manager.release()
        if hasattr(self, 'webrtc_client') and self.webrtc_client.pc:
            asyncio.get_event_loop().run_until_complete(self.webrtc_client.pc.close())
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_P and event.modifiers() & Qt.ControlModifier:
            e_d_func.enable()
            logger.info("Ctrl+P pressed, initiating shutdown...")
            QTimer.singleShot(5000, self.force_close)
            self.close()

    def force_close(self):
        logger.info("Force closing the application")
        # e_d_func.enable()
        QApplication.exit(0)
        

def start_main_app():
    global thread4
    """Function to start the main application threads and window."""
    try:
        thread1 = threading.Thread(target=start_app, name='Thread 1')
        thread4 = threading.Thread(target=screen_s, name='Thread 4', args=(stop_event1,))
        thread5 = threading.Thread(target=join_room, name='Thread 5')
        thread6 = threading.Thread(target=e_d_func.disable, name='Thread 6')
        thread1.start()
        thread4.start()
        thread5.start()
        
        # stop_event1.set()
        thread4.join()
        
        # thread6.start()
        thread1.join()
        thread5.join()
        
        # thread6.start()
    except Exception as e:
        logger.error(f"An error occurred in the main thread: {e}")
        traceback.print_exc()
    # finally:
    #     # Ensure all asyncio tasks are done
    #     loop = asyncio.get_event_loop()
    #     if not loop.is_closed():
    #         pending = asyncio.all_tasks(loop)
    #         loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    #         loop.close()
def start_app():
    """Function to initialize and start the QApplication."""
    try:
        app = QApplication(sys.argv)
        window = FullScreenWindow()
        window.showFullScreen()
        app.exec_()
    except Exception as e:
        logger.error(f"An error occurred in the Qt application: {e}")
        traceback.print_exc()
    finally:
        # Ensure all asyncio tasks are done
        pending = asyncio.all_tasks()
        for task in pending:
            task.cancel()
        asyncio.get_event_loop().run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        asyncio.get_event_loop().close()

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        disclaimer_dialog = DisclaimerDialog()

        if disclaimer_dialog.exec_() == QDialog.Accepted:
            # If the user clicks "Agree", start the main application
            start_main_app()
        else:
            logger.info("User did not accept the disclaimer. Exiting the application.")
            sys.exit(0)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        traceback.print_exc()