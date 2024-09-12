

import socketio
# print(socketio.__version__)
import pyaudio
import logging
import numpy as np
import time
import threading

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
sio1 = socketio.Client(logger=True, engineio_logger=True)

class AudioClient:
    def __init__(self, server_url='https://dash.intellirecruit.ai'):
        self.sio1 = socketio.Client(logger=True, engineio_logger=True)
        self.server_url = server_url
        self.running = False
        self.reconnecting = False

        # Audio settings
        self.FORMAT = pyaudio.paInt32
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 200000

        self.audio = pyaudio.PyAudio()
        self.stream_in = None
        self.stream_out = None

        # Set up socket events
        self.sio1.on('connect_a', self.on_connect)
        self.sio1.on('disconnect_a', self.on_disconnect)
        self.sio1.on('audio', self.on_audio)

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
                print("stream_out")
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
                self.sio1.emit('audio', data)
            except Exception as e:
                logger.error(f"Error capturing or sending audio: {e}")
                time.sleep(0.1)

    def reconnect(self):
        logger.info("Attempting to reconnect...")
        while self.running and not self.sio1.connected:
            try:
                self.sio1.connect(self.server_url)
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
                                             frames_per_buffer=2000)
            
            self.stream_out = self.audio.open(format=self.FORMAT, channels=self.CHANNELS,
                                              rate=self.RATE, output=True,
                                              frames_per_buffer=self.CHUNK)

            self.sio1.connect(self.server_url)
            
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
            if self.sio1.connected:
                self.sio1.disconnect()


if __name__ == "__main__":
    try:
        def loop():
            client = AudioClient()
            
            thread5 = threading.Thread(target=client.run, name='Thread 5')
            thread5.start()
            thread5.join()
            loop()
        loop()
    except Exception as e:
        pass
