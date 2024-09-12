import asyncio
import json
import cv2
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
import aiohttp
import mss
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
# if _name_ == "_main_":
#     # Define the room URL
    

#     # Join the existing room and start the video conference using Selenium
    # join_room(room_url, user_name="interviewee")


logging.basicConfig(level=logging.INFO)

# class ScreenShareTrack(VideoStreamTrack):
#     def _init_(self):
#         super()._init_()
#         self.restart()

#     def restart(self):
#         self.sct = mss.mss()
#         self.monitor = self.sct.monitors[0]  # Capture the primary monitor
#         self.frame_count = 0

#     async def recv(self):
#         self.frame_count += 1
#         if self.frame_count % 30 == 0:  # Log every 30 frames
#             logging.info(f"Sending frame {self.frame_count}")
#         screenshot = self.sct.grab(self.monitor)
#         frame = np.array(screenshot)
#         frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
#         video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
#         pts, time_base = await self.next_timestamp()
#         video_frame.pts = pts
#         video_frame.time_base = time_base
#         return video_frame

# async def reset_room(signaling_url):
#     async with aiohttp.ClientSession() as session:
#         async with session.post(f"{signaling_url}/reset_room/test_room") as resp:
#             if resp.status != 200:
#                 logging.error(f"Failed to reset room: {resp.status}")
                
                
# async def check_stop_signal(signaling_url):
#     try:
#         async with aiohttp.ClientSession() as session:
#             async with session.get(f"{signaling_url}/get_stop_signal/test_room") as resp:
#                 if resp.status == 200:
#                     stop_signal = await resp.json()
                    
#                     return stop_signal.get('stop', False)
#     except Exception as e:
#         logging.error(f"Error checking stop signal: {e}")
#     return False

# async def ensure_signaling(pc, signaling_url, max_retries=5):
#     for attempt in range(max_retries):
#         try:
#             offer = await pc.createOffer()
#             await pc.setLocalDescription(offer)
            
#             offer_dict = object_to_string(pc.localDescription)
            
#             async with aiohttp.ClientSession() as session:
#                 async with session.post(
#                     f"{signaling_url}/offer",
#                     json={"offer": offer_dict, "room": "test_room"}
#                 ) as resp:
#                     if resp.status != 200:
#                         raise Exception(f"Failed to send offer: {resp.status}")
            
#             logging.info("Offer sent to signaling server")
            
#             while True:
#                 if await check_stop_signal(signaling_url):
#                     logging.info("Received stop signal from server")
                    
#                     await reset_room(signaling_url)
#                     await reset_peer_connection(pc)
#                     await main()
#                     # await pc.close()
#                     # return "STOP"

#                 async with aiohttp.ClientSession() as session:
#                     async with session.get(f"{signaling_url}/get_answer/test_room") as resp:
#                         if resp.status == 200:
#                             answer_dict = await resp.json()
#                             answer = object_from_string(json.dumps(answer_dict))
#                             await pc.setRemoteDescription(answer)
#                             logging.info("Answer received and set")
#                             return "CONNECTED"
#                         elif resp.status != 404:
#                             raise Exception(f"Unexpected status when getting answer: {resp.status}")
#                 await asyncio.sleep(1)
#         except Exception as e:
#             logging.error(f"Signaling attempt {attempt + 1} failed: {e}")
#             if attempt == max_retries - 1:
#                 raise
#         await asyncio.sleep(2 ** attempt)  # Exponential backoff

# async def reset_peer_connection(pc):
#     if pc:
#         await pc.close()
#     new_pc = RTCPeerConnection()
    
#     @new_pc.on("track")
#     def on_track(track):
#         logger.info("Receiving video track")
        
#         async def display_video():
#             global screen_sharing_frame
#             frame_count = 0
#             while True:
#                 try:
#                     frame = await track.recv()
#                     frame_count += 1
#                     if frame_count % 30 == 0:  # Log every 30 frames
#                         logger.info(f"Received frame {frame_count}")
                    
#                     img = frame.to_ndarray(format="rgb24")
                    
#                     with frame_lock:
#                         screen_sharing_frame = img
#                 except Exception as e:
#                     logger.error(f"Error in display_video: {e}")
#                     await asyncio.sleep(5)
        
#         asyncio.ensure_future(display_video())

#     @new_pc.on("connectionstatechange")
#     async def on_connectionstatechange():
#         logger.info(f"Connection state is {new_pc.connectionState}")
#         if new_pc.connectionState == "connected":
#             logger.info("Peer connection established!")
#         elif new_pc.connectionState in ["failed", "disconnected", "closed"]:
#             logger.warning(f"Connection state changed to {new_pc.connectionState}")

#     return new_pc

# async def heartbeat(pc, reset_func):
#     while True:
#         if pc.connectionState in ["failed", "disconnected", "closed"]:
#             logging.info("Connection lost. Attempting to reconnect...")
#             return  # Exit heartbeat, main loop will handle reconnection
#         await asyncio.sleep(5)  # Check every 5 seconds

# async def restart_ice(pc):
#     logging.info("Restarting ICE")
#     offer = await pc.createOffer()
#     await pc.setLocalDescription(offer)

# async def main():
#     signaling_url = "https://dash.intellirecruit.ai"  # Update this to your signaling server URL
#     pc = None

#     while True:
#         try:
#             import requests, time
#             requests.post('https://dash.intellirecruit.ai/stop_client/test_room')
            
#             time.sleep(5)
#             await reset_room(signaling_url)
#             pc = await reset_peer_connection(pc)
#             await restart_ice(pc)
#             result = await ensure_signaling(pc, signaling_url)
#             if result == "STOP":
#                 logging.info("Stopping the client as requested by the server")
#                 break
#             logging.info("Screen sharing started")
            
#             while True:
#                 if await check_stop_signal(signaling_url):
                    
#                     logging.info("Stopping the client as requested by the server")
#                     await reset_room
#                     if pc:
#                         await pc.close()
#                     pc = await reset_peer_connection(None)
#                     await pc.close()
#                     return
#                 await asyncio.sleep(5)  # Check every 5 seconds
#         except Exception as e:
#             logging.error(f"An error occurred: {e}")
#             logging.error(traceback.format_exc())
#             await asyncio.sleep(5)  # Wait before retrying
            
# # if _name_ == "_main_":
# #     try:
# #         asyncio.run(main())
# #     except Exception as e:
# #         print("An error occurred:::::::::::::::::::::::::::::", e)
        
# def screen_s():
#     try:
#         asyncio.run(main())
#     except Exception as e:
#         print("An error occurred:::::::::::::::::::::::::::::", e)
        
     
# import socketio
# import pyautogui
# import base64
import time
import io
from aiortc import RTCIceServer, RTCConfiguration



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
# from PIL import Image
# import pyaudio

# sio = socketio.Client(logger=True, engineio_logger=True)
# sio1 = socketio.Client(logger=True, engineio_logger=True)

async def ensure_signaling(pc, signaling_url, max_retries=5):
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{signaling_url}/get_offer/test_room") as resp:
                    if resp.status == 200:
                        offer_json = await resp.json()
                        logger.info(f"Received offer JSON: {json.dumps(offer_json, indent=2)}")
                        if isinstance(offer_json, str):
                            offer_json = json.loads(offer_json)
                        
                        if isinstance(offer_json, dict) and "sdp" in offer_json and "type" in offer_json:
                            offer = RTCSessionDescription(sdp=offer_json["sdp"], type=offer_json["type"])
                        else:
                            raise Exception(f"Unexpected offer format: {offer_json}")
                        
                        await pc.setRemoteDescription(offer)
                        answer = await pc.createAnswer()
                        await pc.setLocalDescription(answer)
                        
                        answer_dict = {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
                        
                        async with session.post(
                            f"{signaling_url}/answer",
                            json={"answer": answer_dict, "room": "test_room"}
                        ) as resp:
                            if resp.status != 200:
                                raise Exception(f"Failed to send answer: {resp.status}")
                        
                        logger.info("Answer sent to signaling server")
                        return
                    elif resp.status != 404:
                        raise Exception(f"Unexpected status when getting offer: {resp.status}")
            await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Signaling attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
        await asyncio.sleep(2 ** attempt)  # Exponential backoff

async def reset_peer_connection(pc):
    if pc:
        await pc.close()

    
    configuration = RTCConfiguration(iceServers=ice_servers)

    # Then use this configuration when creating the RTCPeerConnection
    new_pc = RTCPeerConnection(configuration=configuration)
    # new_pc = RTCPeerConnection()
    
    @new_pc.on("track")
    def on_track(track):
        logger.info("Receiving video track")
        
        async def display_video():
            global screen_sharing_frame
            frame_count = 0
            while True:
                try:
                    frame = await track.recv()
                    frame_count += 1
                    if frame_count % 30 == 0:  # Log every 30 frames
                        logger.info(f"Received frame {frame_count}")
                    
                    img = frame.to_ndarray(format="rgb24")
                    print(img.shape)
                    original_height, original_width, _ = img.shape
                    print(f"Original image shape: {img.shape}")
                    img = img[:, 0:-700]
                    with frame_lock:
                        screen_sharing_frame = img.tobytes()
                except Exception as e:
                    logger.error(f"Error in display_video: {e}")
                    await asyncio.sleep(5)
                    import requests
                    # requests.post('https://dash.intellirecruit.ai/stop_client/test_room')
                    # print("signal sent to stop client...")
                    # import time
                    # print("resetting_room..")
                    # time.sleep(5)
                    
                    # await reset_room("https://dash.intellirecruit.ai")
                    # print("resetting peer connection")
                    # time.sleep(5)
                    
                    # pc = await reset_peer_connection(None)
                    # print("closing pc..")
                    # time.sleep(5)
                    
                    # pc.close()
                    # time.sleep(5)
                    # asyncio.ensure_future(display_video())



        asyncio.ensure_future(display_video())

    @new_pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state is {new_pc.connectionState}")
        if new_pc.connectionState == "connected":
            logger.info("Peer connection established!")
        elif new_pc.connectionState in ["failed", "disconnected", "closed"]:
            logger.warning(f"Connection state changed to {new_pc.connectionState}")

    return new_pc

async def heartbeat(pc, reset_func):
    while True:
        if pc.connectionState in ["failed", "disconnected", "closed"]:
            logger.info("Connection lost. Attempting to reconnect...")
            return  # Exit heartbeat, main loop will handle reconnection
        await asyncio.sleep(5)  # Check every 5 seconds

async def reset_room(signaling_url):
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{signaling_url}/reset_room/test_room") as resp:
            if resp.status != 200:
                logger.error(f"Failed to reset room: {resp.status}")

async def check_stop_signal(signaling_url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{signaling_url}/get_stop_signal/test_room") as resp:
                if resp.status == 200:
                    stop_signal = await resp.json()
                    return stop_signal.get('stop', False)
    except Exception as e:
        logger.error(f"Error checking stop signal: {e}")
    return False

async def start_screen_sharing_client(pc):
    SIGNALING_URL = "https://dash.intellirecruit.ai"
    while True:
        try:
            import requests
            if requests.get('https://dash.intellirecruit.ai/has_offer_global/').json()['has_offer'] == False:
                # Instead of restarting the script, we'll reset everything
                print("Resetting and reinitializing...")
                requests.post('https://dash.intellirecruit.ai/stop_client/test_room')
                print("signal sent to stop client...")
                import time
                time.sleep(5)
                
                await reset_room("https://dash.intellirecruit.ai")
                time.sleep(5)
                pc = await reset_peer_connection(None)
                time.sleep(5)
                pc.close()
                time.sleep(5)
                # Recreate peer connection
                if pc:
                    await pc.close()
                    time.sleep(5)
                pc = await reset_peer_connection(None)
                
                # Wait for a bit to allow the server to reset
                await asyncio.sleep(5)
                await start_screen_sharing_client(pc)

            if not pc:
                pc = await reset_peer_connection(pc)
            
            await ensure_signaling(pc, SIGNALING_URL)
            logger.info("Waiting for video...")
            
            heartbeat_task = asyncio.create_task(heartbeat(pc, reset_peer_connection))
            
            while True:
                if await check_stop_signal(SIGNALING_URL):
                    logger.info("Stopping the client...")
                    await reset_room(SIGNALING_URL)
                    if pc:
                        await pc.close()
                    pc = await reset_peer_connection(pc)
                    break

                await asyncio.sleep(0)  # Allow other tasks to run
                
                if heartbeat_task.done():
                    break  # Exit inner loop to reconnect

        except Exception as e:
            logger.error(f"An error occurred: {e}")
            logger.error(traceback.format_exc())
            await asyncio.sleep(5)  # Wait before retrying
            await reset_room(SIGNALING_URL)


    
    
    

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
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QLabel, QTextEdit, QScrollArea
from PyQt5.QtCore import Qt, QTimer, pyqtSlot,  pyqtSignal, QObject, QRunnable, QThreadPool
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
from PyQt5.Qsci import QsciScintilla, QsciLexerPython
import threading
import elevate
import sys
import numpy as np
import traceback
import time
# import e_d_func
# import try1
from threading import Lock
# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(_name_)

SERVER_URL = "wss://dashboard.intellirecruit.ai/websocket"
CLIENT_ID = "client2"
screen_sharing_frame = None
frame_lock = Lock()

def custom_excepthook(exc_type, exc_value, exc_traceback):
    # Log the exception
    print("Uncaught exception", exc_type, exc_value)
    traceback.print_tb(exc_traceback)
    
    # Call the default excepthook
    sys._excepthook_(exc_type, exc_value, exc_traceback)

sys.excepthook = custom_excepthook

class CameraManager:
    def _init_(self, max_retries=5):
        self.cap = None
        self.lock = threading.Lock()
        self.initialize_camera(max_retries)

    def initialize_camera(self, max_retries):
        for attempt in range(max_retries):
            try:
                self.cap = cv2.VideoCapture(0)
                if self.cap.isOpened():
                    logger.info(f"Camera initialized successfully on attempt {attempt + 1}")
                    return
                else:
                    logger.warning(f"Failed to open camera on attempt {attempt + 1}")
            except Exception as e:
                logger.error(f"Error initializing camera on attempt {attempt + 1}: {e}")
            time.sleep(1)  # Wait before retrying
        logger.error("Failed to initialize camera after maximum retries")

    def read_frame(self):
        if self.cap is None or not self.cap.isOpened():
            logger.warning("Camera is not initialized or opened")
            return None
        with self.lock:
            ret, frame = self.cap.read()
        if ret:
            return frame
        logger.warning("Failed to read frame from camera")
        return None

    def release(self):
        if self.cap:
            self.cap.release()

camera_manager = CameraManager()


class VideoTransformTrack(VideoStreamTrack):
    def _init_(self):
        super()._init_()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            print("Could not start video capture")
            raise RuntimeError("Could not start video capture")

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        frame = camera_manager.read_frame()
        if frame is None:
            if self.last_frame is None:
                # If we've never successfully read a frame, create a blank one
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                frame = self.last_frame
            logger.warning(f"Using fallback frame (count: {self.frame_count})")
        else:
            self.last_frame = frame

        # self.frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base

        return video_frame

async def process_incoming_video(track):
    global incoming_frame
    try:
        while True:
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")
            with frame_lock:
                incoming_frame = img
    except Exception as e:
        print(f"Error in process_incoming_video: {e}, stopping video processing.")


class WebRTCClient(QObject):
    incoming_frame = pyqtSignal(np.ndarray)
    outgoing_frame = pyqtSignal(np.ndarray)

    def _init_(self, running_event):
        super()._init_()
        self.pc = None
        self.video_track = None
        self.running_event = running_event

    async def run_offer(self,pc):
        await pc.setLocalDescription(await pc.createOffer())
        print(f"Local Description: {pc.localDescription.sdp}")
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    async def run_answer(self, pc, offer):
        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer["sdp"], type=offer["type"]))
        print(f"Remote Description set with SDP: {offer['sdp']}")
        await pc.setLocalDescription(await pc.createAnswer())
        print(f"Local Description (Answer): {pc.localDescription.sdp}")
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


    async def consume_signaling(self,pc, websocket):
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
                video_track.cap.release()
                
                print("Resources released and connections closed")

            # Ensure Pygame window is closed
            # stop_event.set()
            # if display_thread:
            #     await display_thread
            # stop_event.clear()
                break

    async def main_webrtc(self, running_event):
        while running_event.is_set():
            global video_track
            global display_thread
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

            # Then use this configuration when creating the RTCPeerConnection
            self.pc = RTCPeerConnection(configuration=configuration)# pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[ice_server]))
    
            self.video_track = VideoTransformTrack()
            self.pc.addTrack(self.video_track)

            @self.pc.on("track")
            def on_track(track):
                logger.info("Receiving video track")
                if track.kind == "video":
                    recorder.addTrack(track)
                    asyncio.ensure_future(process_incoming_video(track))
                    
                    async def display_incoming_video():
                        import aiortc
                        while running_event.is_set():
                            try:
                                frame = await track.recv()
                                img = frame.to_ndarray(format="bgr24")
                                self.incoming_frame.emit(img)
                            except aiortc.mediastreams.MediaStreamError:
                                logger.warning("MediaStreamError occurred. Attempting to reconnect...")
                                await asyncio.sleep(1)  # Wait a bit before trying again
                                continue  # Continue the loop to try receiving again
                            except Exception as e:
                                logger.error(f"Unexpected error in display_incoming_video: {e}")
                                break  # Exit the loop if an unexpected error occurs

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

                    # while running_event.is_set():
                    #     await asyncio.sleep(1)

            except websockets.exceptions.ConnectionClosed as e:
                logger.error(f"WebSocket connection closed unexpectedly: {e}")
            except Exception as e:
                logger.error(f"An error occurred: {e}")
            finally:
                logger.info("Cleaning up...")
                await recorder.stop()
                await self.pc.close()
            await asyncio.sleep(5)  # Wait for 5 seconds before restarting
            continue

class WebRTCWorker(QRunnable):
    def _init_(self, webrtc_client):
        super()._init_()
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
    def _init_(self, running_event):
        super()._init_()
        self.running_event = running_event

    def run(self):
        try:
            main_vad(self.running_event)
        except Exception as e:
            error_msg = f"An error occurred in the VAD thread: {e}\n{traceback.format_exc()}"
            logger.error(error_msg)



def screen_sharing_worker():
    asyncio.run(start_screen_sharing_client(None))

def start_screen_sharing_thread():
    global screen_sharing_thread
    screen_sharing_thread = threading.Thread(target=screen_sharing_worker)
    screen_sharing_thread.start()
    return screen_sharing_thread


def check_and_switch_camera(self):
    available_cameras = []
    index = 0
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        available_cameras.append(index)
        cap.release()
        index += 1
    
    logger.info(f"Available cameras: {available_cameras}")
    
    if len(available_cameras) > 1:
        current_index = available_cameras.index(camera_manager.cap.get(cv2.CAP_PROP_POS_FRAMES))
        next_index = available_cameras[(current_index + 1) % len(available_cameras)]
        camera_manager.release()
        camera_manager.initialize_camera(max_retries=3, camera_index=next_index)
        logger.info(f"Switched to camera index {next_index}")
    else:
        logger.warning("No alternative cameras available")

class FullScreenWindow(QMainWindow):
    def _init_(self):
        super()._init_()
        
        self.running_event = threading.Event()
        self.running_event.set()
        start_screen_sharing_thread()
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
        
        left_widget = QWidget(self)
        left_layout = QVBoxLayout(left_widget)
        
        self.screen_sharing_widget = QWidget(self)
        screen_sharing_layout = QVBoxLayout(self.screen_sharing_widget)

        self.screen_sharing_label = QLabel(self)
        self.screen_sharing_label.setAlignment(Qt.AlignCenter)
        self.screen_sharing_label.setText("Screen Sharing")
        self.screen_sharing_label.setStyleSheet("""
            background-color: black;
            color: white;
            border: 1px solid #555;
        """)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidget(self.screen_sharing_label)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: black;
                border: none;
            }
            QScrollBar {
                background-color: #333;
            }
            QScrollBar::handle {
                background-color: #666;
            }
        """)
        screen_sharing_layout.addWidget(self.scroll_area)
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        # Add the screen sharing widget to the main layout
        main_layout.addWidget(self.screen_sharing_widget, 3)
        
        right_layout = QVBoxLayout()
        self.setStyleSheet("""
        QMainWindow {
            background-color: #1e1e1e;
        }
        QLabel {
            background-color: #2e2e2e;
            border: 1px solid #555;
        }
        QScrollArea {
            border: none;
        }
    """)
        
        self.outgoing_camera_label = QLabel(self)
        self.outgoing_camera_label.setStyleSheet("background-color: #2e2e2e; border: 1px solid #555;")
        right_layout.addWidget(self.outgoing_camera_label)

        self.incoming_camera_label = QLabel(self)
        self.incoming_camera_label.setStyleSheet("background-color: #2e2e2e; border: 1px solid #555;")
        right_layout.addWidget(self.incoming_camera_label)

        self.warning_label = QLabel(self)
        self.warning_label.setStyleSheet("color: #ff6b6b; font-size: 26px;")
        right_layout.addWidget(self.warning_label)
        
        main_layout.addLayout(right_layout)

        # Add logo here
        self.logo_label = QLabel()
        logo_pixmap = QPixmap("sony.jpeg")  # Replace with the path to your logo image
        scaled_logo = logo_pixmap.scaled(100, 50, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.logo_label.setPixmap(scaled_logo)
        bottom_layout.addWidget(self.logo_label)
        # End of logo addition

        self.disclaimer_label = QLabel("Disclaimer: This screen sharing session is confidential and proprietary. Unauthorized recording or distribution is prohibited.")
        self.disclaimer_label.setStyleSheet("""
            color: #888888;
            font-size: 15px;
        """)
        self.disclaimer_label.setWordWrap(True)
        bottom_layout.addWidget(self.disclaimer_label, 1)  # The '1' allows it to expand horizontally

        screen_sharing_layout.addWidget(bottom_widget)


        self.disclaimer_label1 = QLabel("© Copyright, IntelliRecruit 2024-25, all rights reserved.")
        self.disclaimer_label1.setStyleSheet("""
            color: #888888;
            font-size: 15px;
        """)
        self.disclaimer_label1.setWordWrap(True)
        bottom_layout.addWidget(self.disclaimer_label1, 1)  # The '1' allows it to expand horizontally

        screen_sharing_layout.addWidget(bottom_widget)

        # Add the screen sharing widget to the main layout
        main_layout.addWidget(self.screen_sharing_widget, 3)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_outgoing_frame)
        self.timer.start(100)  # Update the frame every 30 ms

        self.webrtc_client = WebRTCClient(self.running_event)
        self.webrtc_client.incoming_frame.connect(self.update_incoming_frame)
        
        self.thread_pool = QThreadPool()
        self.webrtc_worker = WebRTCWorker(self.webrtc_client)
        self.thread_pool.start(self.webrtc_worker)

        self.vad_worker = VADWorker(self.running_event)
        self.thread_pool.start(self.vad_worker)
        
        self.screen_sharing_timer = QTimer(self)
        self.screen_sharing_timer.timeout.connect(self.update_screen_sharing)
        self.screen_sharing_timer.start(33)  # Update every ~33ms (30 FPS)

        self.screen_sharing_thread = None
        self.video_client_thread = None

    def zoom_in(self):
        self.zoom_factor *= 1.2
        self.update_screen_sharing()

    def zoom_out(self):
        self.zoom_factor /= 1.2
        self.update_screen_sharing()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_screen_sharing()

    def update_screen_sharing(self):
        global screen_sharing_frame
        with frame_lock:
            if screen_sharing_frame is not None:
                qimage = QImage(screen_sharing_frame, 1220, 1080, 
                                QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimage)
                
                # Scale the pixmap to fit the width of the scroll area, maintaining aspect ratio
                available_width = self.scroll_area.width() - 2  # Subtract 2 for border
                scaled_pixmap = pixmap.scaledToWidth(available_width, Qt.SmoothTransformation)
                
                self.screen_sharing_label.setPixmap(scaled_pixmap)
                self.screen_sharing_label.setFixedSize(scaled_pixmap.size())    #     self.code_editor.setUtf8(True)
  
    @pyqtSlot()
    def update_outgoing_frame(self):
        try:
            frame = camera_manager.read_frame()
            if frame is not None:
                image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                self.outgoing_camera_label.setPixmap(QPixmap.fromImage(image))
                self.webrtc_client.outgoing_frame.emit(frame)
            else:
                self.handle_frame_failure()
        except Exception as e:
            logger.error(f"Error in update_outgoing_frame: {e}")
            self.handle_frame_failure()

    def handle_frame_failure(self):
        # Display a placeholder image or message
        placeholder = QPixmap(640, 480)
        placeholder.fill(Qt.black)
        self.outgoing_camera_label.setPixmap(placeholder)
        self.warning_label.setText("Camera feed unavailable")
        
        # Attempt to reinitialize the camera
        self.reinitialize_camera()

    def reinitialize_camera(self):
        camera_manager.release()
        camera_manager.initialize_camera(max_retries=3)


    @pyqtSlot(np.ndarray)
    def update_incoming_frame(self, frame):
        image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        self.incoming_camera_label.setPixmap(QPixmap.fromImage(image))

    def closeEvent(self, event):
        logger.info("Closing application...")
        self.running_event.clear()
        self.timer.stop()
        self.thread_pool.waitForDone(5000)  # Wait up to 5 seconds for threads to finish
        camera_manager.release()
        if hasattr(self, 'webrtc_client') and self.webrtc_client.pc:
            asyncio.get_event_loop().run_until_complete(self.webrtc_client.pc.close())
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_P and event.modifiers() & Qt.ControlModifier:
            screen_sharing_thread.join()
            logger.info("Ctrl+P pressed, initiating shutdown...")
            QTimer.singleShot(5000, self.force_close)  # Force close after 5 seconds
            self.close()
            # e_d_func.enable()

    def force_close(self):
        logger.info("Force closing the application")
        QApplication.exit(0)
        # e_d_func.enable()

def start_app():
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

  
import ctypes
def hide_console():
    """Hide the console window."""
    ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)


if _name_ == "_main_":
    try:
        # if not ctypes.windll.shell32.IsUserAnAdmin():
        elevate.elevate()
            # sys.exit()  # Exit the script to prevent the non-elevated version from continuing

    # Hide the console window after elevation
        # hide_console()
       

       
        thread1 = threading.Thread(target=start_app, name='Thread 1')
     
        
        # thread4 = threading.Thread(target=screen_s, name='Thread 4')
        thread5 = threading.Thread(target=join_room, name='Thread 5')
        thread1.start()
      
        # thread4.start()
        thread5.start()
        # thread5.join()
        # # thread5.join()
        thread1.join()
        # thread4.join()
        thread5.join()
        # thread2.join()
        # thread3.join()
        # thread4.join()
    except Exception as e:
        logger.error(f"An error occurred in the main thread: {e}")
        traceback.print_exc()