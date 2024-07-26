

import socketio
import pyautogui
import base64
import time
import io
from PIL import Image
import pyaudio

sio = socketio.Client(logger=True, engineio_logger=True)
sio1 = socketio.Client(logger=True, engineio_logger=True)



    
    
    

import sys
import io, os
from contextlib import redirect_stdout, redirect_stderr
import argparse
import io
import speech_recognition as sr
import torch
import asyncio
import json
import cv2
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaRecorder
import websockets
from av import VideoFrame
import logging
from datetime import datetime
from queue import Queue
from tempfile import NamedTemporaryFile
from sys import platform
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QLabel, QTextEdit
from PyQt5.QtCore import Qt, QTimer, pyqtSlot, QThread, pyqtSignal, QObject, QRunnable, QThreadPool
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
from PyQt5.Qsci import QsciScintilla, QsciLexerPython
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
            
        if ret:
            return frame
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
        
            self.pc = RTCPeerConnection()
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

class FullScreenWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.running_event = threading.Event()
        self.running_event.set()

        self.setWindowTitle("Kiosk App")
        # self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.WindowDoesNotAcceptFocus)
        # self.setFixedSize(1920, 1080)
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                border: 1px solid #555;
            }
        """)
        
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        left_layout = QVBoxLayout()
        
        self.code_editor = QsciScintilla(self)
        self.setup_editor()
        left_layout.addWidget(self.code_editor)
        
        self.output_display = QTextEdit(self)
        self.output_display.setReadOnly(True)
        self.output_display.setStyleSheet("""
            QTextEdit {
                background-color: #2e2e2e;
                color: #ffffff;
                border: 1px solid #555;
            }
        """)
        left_layout.addWidget(self.output_display)
        left_layout.setStretch(0, 2)
        left_layout.setStretch(1, 1)
        
        main_layout.addLayout(left_layout)
        
        right_layout = QVBoxLayout()
        
        self.run_button = QPushButton("Run Code", self)
        self.run_button.clicked.connect(self.run_code)
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #3a3a3a;
                color: #ffffff;
                border: 1px solid #555;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
            QPushButton:pressed {
                background-color: #2a2a2a;
            }
        """)
        right_layout.addWidget(self.run_button)
        
        self.outgoing_camera_label = QLabel(self)
        self.outgoing_camera_label.setStyleSheet("background-color: #2e2e2e; border: 1px solid #555;")
        right_layout.addWidget(self.outgoing_camera_label)

        self.incoming_camera_label = QLabel(self)
        self.incoming_camera_label.setStyleSheet("background-color: #2e2e2e; border: 1px solid #555;")
        right_layout.addWidget(self.incoming_camera_label)

        self.warning_label = QLabel(self)
        self.warning_label.setStyleSheet("color: #ff6b6b; font-size: 16px;")
        right_layout.addWidget(self.warning_label)
        
        main_layout.addLayout(right_layout)

        # Add logo here
        self.logo_label = QLabel(self)
        logo_pixmap = QPixmap("sony.jpeg")  # Replace with the path to your logo image
        self.logo_label.setPixmap(logo_pixmap)
        right_layout.addWidget(self.logo_label)
        # End of logo addition

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
        
        lexer = QsciLexerPython()
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
        
        self.code_editor.setLexer(lexer)

    
    def run_code(self):
        code = self.code_editor.text()
        
        # Create string buffers to capture stdout and stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        try:
            # Create a dictionary to serve as a local namespace for exec
            local_vars = {}
            
            # Redirect stdout and stderr to our buffers
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(code, globals(), local_vars)
            
            # Get the captured output
            output = stdout_buffer.getvalue()
            error_output = stderr_buffer.getvalue()
            
            # Check if there's any output or if any variables were created/modified
            if output:
                self.output_display.setText(f"Output:\n{output}")
            elif error_output:
                self.output_display.setText(f"Errors:\n{error_output}")
            elif local_vars:
                # If no output but variables were created/modified, show them
                var_output = "\n".join(f"{k} = {v}" for k, v in local_vars.items() if not k.startswith("__"))
                self.output_display.setText(f"No print output. Variables created/modified:\n{var_output}")
            else:
                self.output_display.setText("Code executed successfully (no output or variables modified).")
            
            # Log for debugging
            print(f"Stdout: {output}")
            print(f"Stderr: {error_output}")
            print(f"Local vars: {local_vars}")
            
        except Exception as e:
            error_msg = f"Error executing code: {str(e)}\n\n{traceback.format_exc()}"
            self.output_display.setText(error_msg)
            logger.error(error_msg)
        finally:
            # Close the buffers
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
            logger.info("Ctrl+P pressed, initiating shutdown...")
            QTimer.singleShot(5000, self.force_close)  # Force close after 5 seconds
            self.close()
            e_d_func.enable()

    def force_close(self):
        logger.info("Force closing the application")
        QApplication.exit(0)
        e_d_func.enable()

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

  

if __name__ == "__main__":
    try:
        elevate.elevate()
        # start_app()
  
    # try:
       
       
        thread1 = threading.Thread(target=start_app, name='Thread 1')
 
        
        # # thread5.start()
        thread1.start()
      
        
        # # thread5.join()
        thread1.join()
        # thread2.join()
        # thread3.join()
        # thread4.join()
    except Exception as e:
        logger.error(f"An error occurred in the main thread: {e}")
        traceback.print_exc()
