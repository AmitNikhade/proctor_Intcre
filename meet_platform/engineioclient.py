import engineio
import pyaudio
import logging
import numpy as np
import time
import threading

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
eio1 = engineio.Client(logger=True)

class AudioClient:
    def __init__(self, server_url='udp://dash.intellirecruit.ai:1234'):
        self.eio1 = engineio.Client(logger=True)
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
        self.eio1.on('connect', self.on_connect)
        self.eio1.on('disconnect', self.on_disconnect)
        self.eio1.on('audio', self.on_audio)

    def on_connect(self):
        print("Connected to server")
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
            if self.stream_out and not self.stream_out.is_stopped():
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
                self.eio1.emit('audio', data)
            except Exception as e:
                logger.error(f"Error capturing or sending audio: {e}")
                time.sleep(0.1)

    def reconnect(self):
        logger.info("Attempting to reconnect...")
        while self.running and not self.eio1.connected:
            try:
                self.eio1.connect(self.server_url)
                logger.info("Reconnected successfully")
                break
            except Exception as e:
                logger.error(f"Reconnection failed: {e}")
                time.sleep(5)  # Wait for 5 seconds before trying again

    def run(self):
        self.running = True
        try:
            self.eio1.connect(self.server_url)

            self.stream_in = self.audio.open(format=self.FORMAT, channels=self.CHANNELS,
                                             rate=self.RATE, input=True,
                                             frames_per_buffer=2000)
            
            self.stream_out = self.audio.open(format=self.FORMAT, channels=self.CHANNELS,
                                              rate=self.RATE, output=True,
                                              frames_per_buffer=self.CHUNK)
            
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
            if self.eio1.connected:
                self.eio1.disconnect()

if __name__ == "__main__":
    try:
        client = AudioClient()
        client.run()
    except Exception as e:
        pass