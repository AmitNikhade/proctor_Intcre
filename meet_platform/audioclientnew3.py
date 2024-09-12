import asyncio
import json
import aiohttp  # For HTTP requests
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from speech_recognition import Recognizer, Microphone

class MicrophoneStreamTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self):
        super().__init__()
        self.recognizer = Recognizer()
        self.microphone = Microphone()
        self.audio_generator = self.audio_stream()

    def audio_stream(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            while True:
                audio = self.recognizer.listen(source)
                yield audio.get_raw_data(convert_rate=8000, convert_width=2)

    async def recv(self):
        data = next(self.audio_generator)
        return data  # In a real application, proper conversion to the right audio frame format is necessary

async def connect_to_server(pc):
    async with aiohttp.ClientSession() as session:
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        try:
            async with session.post('https://dash.intellirecruit.ai', json={
                'sdp': pc.localDescription.sdp,
                'type': pc.localDescription.type
            }) as resp:
                if resp.status != 200:
                    content = await resp.text()
                    print(f"Failed to connect to server: {resp.status}, {content}")
                    return  # Exit the function or handle error differently
                answer = await resp.json()
                await pc.setRemoteDescription(RTCSessionDescription(sdp=answer['sdp'], type=answer['type']))
        except Exception as e:
            print(f"Error connecting to server: {e}")

async def main():
    pc = RTCPeerConnection()
    local_track = MicrophoneStreamTrack()
    pc.addTrack(local_track)

    # Handle tracks from the server
    @pc.on("track")
    def on_track(track):
        print(f"Track {track.kind} received from server")

    await connect_to_server(pc)

    # Wait for communication or CTRL+C
    try:
        print("Communication started. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        await pc.close()

if __name__ == "__main__":
    asyncio.run(main())
