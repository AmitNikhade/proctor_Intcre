import asyncio
import json
import cv2
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaRecorder
import websockets
from av import VideoFrame
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

SERVER_URL = "wss://dashboard.intellirecruit.ai/websocket"
CLIENT_ID = "client1"

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
        pts, time_base = await self.next_timestamp()

        ret, frame = self.cap.read()
        if not ret:
            logger.error("Failed to capture frame")
            raise RuntimeError("Failed to capture frame")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base

        # Display the outgoing frame for debugging purposes
        cv2.imshow('Outgoing Video - Client 1', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise RuntimeError("Quitting video display")

        logger.debug("Captured frame")
        return video_frame

async def run_offer(pc):
    await pc.setLocalDescription(await pc.createOffer())
    logger.debug(f"Local Description: {pc.localDescription.sdp}")
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

async def consume_signaling(pc, websocket):
    async for message in websocket:
        logger.info(f"Received message: {message}")
        msg = json.loads(message)
        if msg["type"] == "answer":
            logger.debug(f"Answer SDP: {msg['answer']['sdp']}")
            await pc.setRemoteDescription(RTCSessionDescription(sdp=msg["answer"]["sdp"], type=msg["answer"]["type"]))
        else:
            logger.warning(f"Unexpected message type: {msg['type']}")

async def main():
    recorder = MediaRecorder("received_video_client1.mp4")

    pc = RTCPeerConnection()
    video_track = VideoTransformTrack()
    pc.addTrack(video_track)

    @pc.on("track")
    def on_track(track):
        logger.info("Receiving video track")
        if track.kind == "video":
            # Add incoming video track to recorder and display it
            recorder.addTrack(track)

            async def display_incoming_video():
                while True:
                    frame = await track.recv()
                    img = frame.to_ndarray(format="bgr24")
                    cv2.imshow('Incoming Video - Client 1', img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        raise RuntimeError("Quitting video display")

            asyncio.ensure_future(display_incoming_video())
            logger.debug("Added video track to recorder and displaying incoming video")

    try:
        logger.info(f"Attempting to connect to {SERVER_URL}")
        async with websockets.connect(SERVER_URL) as websocket:
            logger.info("Connected to WebSocket server")

            await websocket.send(json.dumps({"type": "join", "client_id": CLIENT_ID}))
            logger.info(f"Sent join message for {CLIENT_ID}")

            offer = await run_offer(pc)
            await websocket.send(json.dumps({"type": "offer", "offer": offer}))
            logger.info("Sent offer")

            await consume_signaling(pc, websocket)

            await recorder.start()
            logger.info("Recorder started")

            while True:
                await asyncio.sleep(1)

    except websockets.exceptions.ConnectionClosed as e:
        logger.error(f"WebSocket connection closed unexpectedly: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        logger.info("Cleaning up...")
        await recorder.stop()
        await pc.close()
        video_track.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
