

import socketio
import pyautogui
import base64
import time
import io
from PIL import Image

sio = socketio.Client(logger=True, engineio_logger=True)

@sio.event(namespace='/screen')
def connect():
    print('Connection established to /screen namespace')

@sio.event(namespace='/screen')
def connect_error(data):
    print(f"Connection failed: {data}")

@sio.event(namespace='/screen')
def disconnect():
    print('Disconnected from /screen namespace')

def capture_and_send_screen():
    while True:
        if sio.connected:
            try:
                screenshot = pyautogui.screenshot()
                screenshot = screenshot.resize((1280, 720), Image.LANCZOS)
                buffered = io.BytesIO()
                screenshot.save(buffered, format="JPEG", quality=100)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                print(f"Sending image data. Size: {len(img_str)} bytes")
                sio.emit('screen_data', {'image': img_str}, namespace='/screen')
                time.sleep(0.1)  # Adjust as needed
            except Exception as e:
                print(f"Error capturing or sending screen: {e}")
                time.sleep(1)
        else:
            print("Not connected. Waiting...")
            time.sleep(5)

# if __name__ == '__main__':
#     sio.connect('https://dashboard.intellirecruit.ai', namespaces=['/screen'])
#     capture_and_send_screen()
    
    
    
    

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
import try1

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

SERVER_URL = "wss://dashboard.intellirecruit.ai/websocket"
CLIENT_ID = "client1"

def custom_excepthook(exctype, value, tb):
    error_msg = ''.join(traceback.format_exception(exctype, value, tb))
    logger.error(f"Uncaught exception:\n{error_msg}")

sys.excepthook = custom_excepthook

class CameraManager:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.lock = threading.Lock()

    def read_frame(self):
        with self.lock:
            ret, frame = self.cap.read()
        if ret:
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame = try1.process_frame(frame)
            # time.sleep(5)
            return frame
        return None

    def release(self):
        self.cap.release()

camera_manager = CameraManager()

class VideoTransformTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        frame = camera_manager.read_frame()
        if frame is None:
            logger.error("Failed to capture frame")
            raise RuntimeError("Failed to capture frame")

        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base

        logger.debug("Captured and processed frame")
        return video_frame

class WebRTCClient(QObject):
    incoming_frame = pyqtSignal(np.ndarray)
    outgoing_frame = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.pc = None
        self.video_track = None

    async def run_offer(self, pc):
        await pc.setLocalDescription(await pc.createOffer())
        logger.debug(f"Local Description: {pc.localDescription.sdp}")
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    async def consume_signaling(self, pc, websocket):
        async for message in websocket:
            logger.info(f"Received message: {message}")
            msg = json.loads(message)
            if msg["type"] == "answer":
                logger.debug(f"Answer SDP: {msg['answer']['sdp']}")
                await pc.setRemoteDescription(RTCSessionDescription(sdp=msg["answer"]["sdp"], type=msg["answer"]["type"]))
            else:
                logger.warning(f"Unexpected message type: {msg['type']}")

    async def main_webrtc(self, running_event):
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

        try:
            logger.info(f"Attempting to connect to {SERVER_URL}")
            async with websockets.connect(SERVER_URL) as websocket:
                logger.info("Connected to WebSocket server")

                await websocket.send(json.dumps({"type": "join", "client_id": CLIENT_ID}))
                logger.info(f"Sent join message for {CLIENT_ID}")

                offer = await self.run_offer(self.pc)
                await websocket.send(json.dumps({"type": "offer", "offer": offer}))
                logger.info("Sent offer")

                await self.consume_signaling(self.pc, websocket)

                await recorder.start()
                logger.info("Recorder started")

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
    def __init__(self, webrtc_client, running_event):
        super().__init__()
        self.webrtc_client = webrtc_client
        self.running_event = running_event

    def run(self):
        try:
            asyncio.run(self.webrtc_client.main_webrtc(self.running_event))
        except Exception as e:
            error_msg = f"An error occurred in the WebRTC thread: {e}\n{traceback.format_exc()}"
            logger.error(error_msg)
            
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
        self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.WindowDoesNotAcceptFocus)
        self.setFixedSize(1920, 1080)
        
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
        self.timer.start(30)  # Update the frame every 30 ms

        self.webrtc_client = WebRTCClient()
        self.webrtc_client.incoming_frame.connect(self.update_incoming_frame)
        
        self.thread_pool = QThreadPool()
        self.webrtc_worker = WebRTCWorker(self.webrtc_client, self.running_event)
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
        frame = camera_manager.read_frame()
        if frame is not None:
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            self.outgoing_camera_label.setPixmap(QPixmap.fromImage(image))
            self.webrtc_client.outgoing_frame.emit(frame)

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
            asyncio.run(self.webrtc_client.pc.close())
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
        sys.exit(app.exec_())
    except Exception as e:
        logger.error(f"An error occurred in the Qt application: {e}")
        traceback.print_exc()

        
def main_vad(running_event):
    parser = argparse.ArgumentParser()
    parser.add_argument("--energy_threshold", default=1000, help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2, help="How real time the recording is in seconds.", type=float)
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    # Current raw audio bytes.
    last_sample = bytes()
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False
    
    # Important for linux users. 
    # Prevents permanent application hang and crash by using the wrong Microphone
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)

    record_timeout = args.record_timeout
    temp_file = NamedTemporaryFile().name
    
    with source:
        recorder.adjust_for_ambient_noise(source)
        logger.info("Adjusted for ambient noise.")

    def record_callback(_, audio: sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        data = audio.get_raw_data()
        data_queue.put(data)
        logger.debug("Audio data received and added to queue.")

    # Load silero VAD model
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False)
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    
    logger.info("VAD model loaded.")

    # Start the recording process
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)
    logger.info("Recording started.")

    while running_event.is_set():
        try:
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                logger.debug("Processing audio data from queue.")
                # Concatenate our current audio data with the latest audio data.
                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data

                # Use AudioData to convert the raw data to wav data.
                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                # Write wav data to the temporary file as bytes.
                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read())

                # Read the transcription.
                wav = read_audio(temp_file, sampling_rate=16000)
                speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)

                if speech_timestamps:
                    logger.info('Speech Detected!')
                    # Here you can add code to handle the detected speech
                    # For example, you might want to transcribe it or perform some action
                else:
                    logger.info('Silence Detected')

                # Clear the last sample to start fresh
                last_sample = bytes()
            else:
                # Sleep briefly to avoid busy-waiting
                time.sleep(0.1)
        except Exception as e:
            logger.error(f"Error in VAD processing: {e}")
            # Optionally break the loop if there's a critical error
            # break

    # Cleanup code for VAD
    logger.info("VAD thread stopping...")

if __name__ == "__main__":
    try:
        elevate.elevate()
        sio.connect('https://dashboard.intellirecruit.ai', namespaces=['/screen'])
        thread4 = threading.Thread(target=capture_and_send_screen, name='Thread 4')
        
        thread1 = threading.Thread(target=start_app, name='Thread 1')
        thread2 = threading.Thread(target=main_vad, name='Thread 2')
        thread3 = threading.Thread(target=e_d_func.disable, name='Thread 3')
        thread1.start()
        thread2.start()
        thread3.start()
        thread4.start()
        thread1.join()
        thread2.join()
        thread3.join()
        thread4.join()
    except Exception as e:
        logger.error(f"An error occurred in the main thread: {e}")
        traceback.print_exc()
