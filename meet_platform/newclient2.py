import asyncio
import json
import cv2
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaRecorder
import websockets
from av import VideoFrame
import logging
import pygame
from threading import Lock

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

SERVER_URL = "wss://s.intellirecruit.ai/websocket"
CLIENT_ID = "client2"  # Change this to "client2" for the second client

outgoing_frame = None
incoming_frame = None
frame_lock = Lock()

class VideoTransformTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            logger.error("Could not start video capture")
            raise RuntimeError("Could not start video capture")

    async def recv(self):
        global outgoing_frame
        pts, time_base = await self.next_timestamp()

        ret, frame = self.cap.read()
        if not ret:
            logger.error("Failed to capture frame")
            raise RuntimeError("Failed to capture frame")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base

        with frame_lock:
            outgoing_frame = frame_rgb
        logger.debug("Outgoing frame captured")
        return video_frame

async def run_offer(pc):
    await pc.setLocalDescription(await pc.createOffer())
    logger.debug(f"Local Description: {pc.localDescription.sdp}")
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

async def run_answer(pc, offer):
    await pc.setRemoteDescription(RTCSessionDescription(sdp=offer["sdp"], type=offer["type"]))
    logger.info(f"Remote Description set with SDP: {offer['sdp']}")
    await pc.setLocalDescription(await pc.createAnswer())
    logger.info(f"Local Description (Answer): {pc.localDescription.sdp}")
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

async def consume_signaling(pc, websocket):
    async for message in websocket:
        logger.info(f"Received message: {message}")
        msg = json.loads(message)
        if msg["type"] == "offer":
            answer = await run_answer(pc, msg["offer"])
            await websocket.send(json.dumps({"type": "answer", "answer": answer}))
            logger.info("Sent answer")
        elif msg["type"] == "answer":
            await pc.setRemoteDescription(RTCSessionDescription(sdp=msg["answer"]["sdp"], type=msg["answer"]["type"]))
            logger.info("Set remote description")

async def process_incoming_video(track):
    global incoming_frame
    while True:
        frame = await track.recv()
        img = frame.to_ndarray(format="bgr24")
        with frame_lock:
            incoming_frame = img
        logger.debug("Incoming frame processed")

def display_video_pygame():
    pygame.init()
    screen = pygame.display.set_mode((1280, 480))
    pygame.display.set_caption(f"Video Streams - {CLIENT_ID}")
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))  # Clear the screen

        with frame_lock:
            if outgoing_frame is not None:
                outgoing_surface = pygame.surfarray.make_surface(np.rot90(outgoing_frame))
                screen.blit(outgoing_surface, (0, 0))
            if incoming_frame is not None:
                incoming_surface = pygame.surfarray.make_surface(np.rot90(incoming_frame))
                screen.blit(incoming_surface, (640, 0))

        pygame.display.flip()
        clock.tick(30)  # Limit to 30 FPS

    pygame.quit()

async def setup_peer_connection():
    config = RTCConfiguration(
        iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
    )
    pc = RTCPeerConnection(configuration=config)
    video_track = VideoTransformTrack()
    pc.addTrack(video_track)

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            asyncio.ensure_future(process_incoming_video(track))

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state changed to: {pc.connectionState}")
        if pc.connectionState == "failed":
            logger.error("Connection failed")
            await pc.close()

    @pc.on("icecandidateerror")
    def on_icecandidateerror(error):
        logger.error(f"ICE candidate error: {error}")

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"ICE connection state changed to: {pc.iceConnectionState}")
        if pc.iceConnectionState == "failed":
            logger.error("ICE connection failed")
            await pc.close()

    return pc, video_track

async def connection_with_timeout(pc):
    try:
        await asyncio.wait_for(pc.connectionStateChange.wait(), timeout=30)
    except asyncio.TimeoutError:
        logger.error("Connection timed out")
        await pc.close()

async def main():
    while True:
        try:
            pc, video_track = await setup_peer_connection()
            recorder = MediaRecorder(f"received_video_{CLIENT_ID}.mp4")

            async with websockets.connect(SERVER_URL) as websocket:
                await websocket.send(json.dumps({"type": "join", "client_id": CLIENT_ID}))
                logger.info(f"Joined server with client ID: {CLIENT_ID}")

                if CLIENT_ID == "client1":
                    offer = await run_offer(pc)
                    await websocket.send(json.dumps({"type": "offer", "offer": offer}))
                    logger.info("Sent offer")

                signaling_task = asyncio.create_task(consume_signaling(pc, websocket))
                connection_task = asyncio.create_task(connection_with_timeout(pc))
                await recorder.start()
                logger.info("Recorder started")

                # Run the display function in a separate thread
                display_thread = asyncio.to_thread(display_video_pygame)
                
                # Wait for the signaling task, connection task, and display thread to complete
                await asyncio.gather(signaling_task, connection_task, display_thread)

        except websockets.ConnectionClosed as e:
            logger.error(f"WebSocket connection closed unexpectedly: {e}")
        except Exception as e:
            logger.error(f"An error occurred: {e}")
        finally:
            if 'recorder' in locals() and recorder:
                await recorder.stop()
            if 'pc' in locals() and pc:
                await pc.close()
            if 'video_track' in locals() and video_track:
                video_track.cap.release()
            logger.info("Resources released and connections closed")
            await asyncio.sleep(5)  # Wait before reconnecting

if __name__ == "__main__":
    asyncio.run(main())