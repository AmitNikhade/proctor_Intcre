# screen_sender.py

import cv2
import asyncio
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.signaling import TcpSocketSignaling, BYE
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder
import numpy as np

class ScreenShareTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        ret, frame = self.cap.read()

        if not ret:
            raise Exception("Failed to capture screen")

        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return VideoFrame.from_ndarray(frame_rgb, format="rgb24").to_image(pts, time_base)

async def run(pc, signaling):
    await signaling.connect()

    # Send Screen Track
    pc.addTrack(ScreenShareTrack())

    # Connect signaling
    while True:
        obj = await signaling.receive()

        if isinstance(obj, RTCSessionDescription):
            await pc.setRemoteDescription(obj)

            if obj.type == "offer":
                await pc.setLocalDescription(await pc.createAnswer())
                await signaling.send(pc.localDescription)

        elif obj is BYE:
            break

if __name__ == "__main__":
    signaling = TcpSocketSignaling("127.0.0.1", 8080)
    pc = RTCPeerConnection()

    asyncio.run(run(pc, signaling))
