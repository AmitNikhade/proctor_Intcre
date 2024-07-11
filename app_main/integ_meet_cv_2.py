import socketio
import pyaudio
import numpy as np
import threading
import logging
import time

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AudioClient:
    def __init__(self, server_url='https://dash.intellirecruit.ai'):
        self.sio = socketio.Client(logger=True, engineio_logger=True)
        self.server_url = server_url
        self.running = False
        self.reconnecting = False

        # Audio settings
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 8192

        self.audio = pyaudio.PyAudio()
        self.stream_in = None
        self.stream_out = None

        # Set up socket events
        self.sio.on('connect_a', self.on_connect)
        self.sio.on('disconnect_a', self.on_disconnect)
        self.sio.on('audio', self.on_audio)

    def on_connect(self):
        logger.info("Connected to server")
        self.reconnecting = False

    def on_disconnect(self):
        logger.info("Disconnected from server")
        if self.running and not self.reconnecting:
            self.reconnecting = True
            self.reconnect()

    def on_audio(self, data):
        try:
            audio_data = np.frombuffer(data, dtype=np.int16)
            logger.debug(f"Received audio data of length: {len(audio_data)}")
            if self.stream_out:
                self.stream_out.write(audio_data.tobytes())
        except Exception as e:
            logger.error(f"Error processing received audio: {e}")

    def send_audio(self):
        while self.running:
            try:
                if not self.stream_in or self.stream_in.is_stopped():
                    self.stream_in = self.audio.open(format=self.FORMAT, channels=self.CHANNELS,
                                                    rate=self.RATE, input=True,
                                                    frames_per_buffer=self.CHUNK)
                data = self.stream_in.read(self.CHUNK, exception_on_overflow=False)
                logger.debug(f"Sending audio data of length: {len(data)}")
                self.sio.emit('audio', data)
            except Exception as e:
                logger.error(f"Error capturing or sending audio: {e}")
                time.sleep(0.1)

    def reconnect(self):
        logger.info("Attempting to reconnect...")
        while self.running and not self.sio.connected:
            try:
                self.sio.connect(self.server_url)
                logger.info("Reconnected successfully")
                break
            except Exception as e:
                logger.error(f"Reconnection failed: {e}")
                time.sleep(5)  # Wait for 5 seconds before trying again

    def run(self):
        self.running = True
        try:
            self.stream_in = self.audio.open(format=self.FORMAT, channels=self.CHANNELS,
                                             rate=self.RATE, input=True,
                                             frames_per_buffer=self.CHUNK)
            
            self.stream_out = self.audio.open(format=self.FORMAT, channels=self.CHANNELS,
                                              rate=self.RATE, output=True,
                                              frames_per_buffer=self.CHUNK)

            self.sio.connect(self.server_url)
            
            # Start sending audio in a separate thread
            threading.Thread(target=self.send_audio, daemon=True).start()
            
            while self.running:
                time.sleep(1)

        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.running = False
            if self.stream_in:
                self.stream_in.stop_stream()
                self.stream_in.close()
            if self.stream_out:
                self.stream_out.stop_stream()
                self.stream_out.close()
            self.audio.terminate()
            if self.sio.connected:
                self.sio.disconnect()

if __name__ == '__main__':
    client = AudioClient()
    try:
        client.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user, shutting down...")
    finally:
        client.running = False

# # if __name__ == '__main__':
# #     sio.connect('https://dashboard.intellirecruit.ai')
# #     send_audio()


# import socketio
# import pyaudio
# import numpy as np
# import threading
# import logging
# import time

# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# class AudioClient:
#     def __init__(self, server_url='https://dashboard.intellirecruit.ai'):
#         self.sio = socketio.Client(logger=True, engineio_logger=True)
#         self.server_url = server_url
#         self.running = False
#         self.reconnecting = False

#         # Audio settings
#         self.FORMAT = pyaudio.paInt16
#         self.CHANNELS = 1
#         self.RATE = 44100
#         self.CHUNK = 8192

#         self.audio = pyaudio.PyAudio()
#         self.stream_in = None
#         self.stream_out = None

#         # Set up socket events
#         self.sio.on('connect_a', self.on_connect)
#         self.sio.on('disconnect_a', self.on_disconnect)
#         self.sio.on('audio', self.on_audio)

#     def on_connect(self):
#         logger.info("Connected to server")
#         self.reconnecting = False

#     def on_disconnect(self):
#         logger.info("Disconnected from server")
#         if self.running and not self.reconnecting:
#             self.reconnecting = True
#             self.reconnect()

#     def on_audio(self, data):
#         try:
#             audio_data = np.frombuffer(data, dtype=np.int16)
#             logger.debug(f"Received audio data of length: {len(audio_data)}")
#             if self.stream_out:
#                 self.stream_out.write(audio_data.tobytes())
#         except Exception as e:
#             logger.error(f"Error processing received audio: {e}")

#     def send_audio(self):
#         while self.running:
#             try:
#                 if not self.stream_in or self.stream_in.is_stopped():
#                     self.stream_in = self.audio.open(format=self.FORMAT, channels=self.CHANNELS,
#                                                     rate=self.RATE, input=True,
#                                                     frames_per_buffer=self.CHUNK)
#                 data = self.stream_in.read(self.CHUNK, exception_on_overflow=False)
#                 logger.debug(f"Sending audio data of length: {len(data)}")
#                 self.sio.emit('audio', data)
#             except Exception as e:
#                 logger.error(f"Error capturing or sending audio: {e}")
#                 time.sleep(0.1)

#     def reconnect(self):
#         logger.info("Attempting to reconnect...")
#         while self.running and not self.sio.connected:
#             try:
#                 self.sio.connect(self.server_url)
#                 logger.info("Reconnected successfully")
#                 break
#             except Exception as e:
#                 logger.error(f"Reconnection failed: {e}")
#                 time.sleep(5)  # Wait for 5 seconds before trying again

#     def run(self):
#         self.running = True
#         try:
#             self.stream_in = self.audio.open(format=self.FORMAT, channels=self.CHANNELS,
#                                              rate=self.RATE, input=True,
#                                              frames_per_buffer=self.CHUNK)
            
#             self.stream_out = self.audio.open(format=self.FORMAT, channels=self.CHANNELS,
#                                               rate=self.RATE, output=True,
#                                               frames_per_buffer=self.CHUNK)

#             self.sio.connect(self.server_url)
            
#             # Start sending audio in a separate thread
#             threading.Thread(target=self.send_audio, daemon=True).start()
            
#             while self.running:
#                 time.sleep(1)

#         except Exception as e:
#             logger.error(f"Error in main loop: {e}")
#         finally:
#             self.running = False
#             if self.stream_in:
#                 self.stream_in.stop_stream()
#                 self.stream_in.close()
#             if self.stream_out:
#                 self.stream_out.stop_stream()
#                 self.stream_out.close()
#             self.audio.terminate()
#             if self.sio.connected:
#                 self.sio.disconnect()

# def start_client():
#     client = AudioClient()
#     try:
#         client.run()
#     except KeyboardInterrupt:
#         logger.info("Interrupted by user, shutting down...")
#     finally:
#         client.running = False

# if __name__ == '__main__':
#     client_thread = threading.Thread(target=start_client)
#     client_thread.start()
#     client_thread.join()  # Optional: wait for the client thread to finish
