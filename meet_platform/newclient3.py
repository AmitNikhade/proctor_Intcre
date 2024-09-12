

import asyncio
import json
import cv2
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaRecorder
import websockets
from av import VideoFrame
import pygame
from threading import Lock

SERVER_URL = "wss://s.intellirecruit.ai/websocket"
CLIENT_ID = "client1"  # Change this to "client2" for the second client

outgoing_frame = None
incoming_frame = None
frame_lock = Lock()

# Create an event to signal when to stop Pygame
stop_event = asyncio.Event()

class VideoTransformTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            print("Could not start video capture")
            raise RuntimeError("Could not start video capture")

    async def recv(self):
        global outgoing_frame
        pts, time_base = await self.next_timestamp()

        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture frame")
            raise RuntimeError("Failed to capture frame")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base

        with frame_lock:
            outgoing_frame = frame_rgb
        return video_frame

async def run_offer(pc):
    await pc.setLocalDescription(await pc.createOffer())
    print(f"Local Description: {pc.localDescription.sdp}")
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

async def run_answer(pc, offer):
    await pc.setRemoteDescription(RTCSessionDescription(sdp=offer["sdp"], type=offer["type"]))
    print(f"Remote Description set with SDP: {offer['sdp']}")
    await pc.setLocalDescription(await pc.createAnswer())
    print(f"Local Description (Answer): {pc.localDescription.sdp}")
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

async def consume_signaling(pc, websocket):
    async for message in websocket:
        print(f"Received message: {message}")
        msg = json.loads(message)
        if msg["type"] == "offer":
            answer = await run_answer(pc, msg["offer"])
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
            stop_event.set()
            if display_thread:
                await display_thread  # Wait for the display thread to exit
            stop_event.clear()
            break

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

        # Check if the stop event is set
        if stop_event.is_set():
            running = False

    pygame.quit()

async def start_client():
    global display_thread 
    global video_track
    recorder = MediaRecorder(f"received_video_{CLIENT_ID}.mp4")
    from aiortc import RTCIceServer, RTCConfiguration
    
    
    
    
    from aiortc import RTCIceServer, RTCConfiguration

# Replace with your STUN/TURN server credentials

#  [{'url': 'stun:global.stun.twilio.com:3478', 'urls': 'stun:global.stun.twilio.com:3478'}, {'url': 'turn:global.turn.twilio.com:3478?transport=udp', 'username': '78e87c2da25896f0cdf3b49542a0176afe9e9017d82a619907a2eaff9c0ed84e', 'urls': 'turn:global.turn.twilio.com:3478?transport=udp', 'credential': 'LfHgqhGrjZII2h3wBk4iSs/r3wl3AFe3SclZ8v4bfds='}, {'url': 'turn:global.turn.twilio.com:3478?transport=tcp', 'username': '78e87c2da25896f0cdf3b49542a0176afe9e9017d82a619907a2eaff9c0ed84e', 'urls': 'turn:global.turn.twilio.com:3478?transport=tcp', 'credential': 'LfHgqhGrjZII2h3wBk4iSs/r3wl3AFe3SclZ8v4bfds='}, {'url': 'turn:global.turn.twilio.com:443?transport=tcp', 'username': '78e87c2da25896f0cdf3b49542a0176afe9e9017d82a619907a2eaff9c0ed84e', 'urls': 'turn:global.turn.twilio.com:443?transport=tcp', 'credential': 'LfHgqhGrjZII2h3wBk4iSs/r3wl3AFe3SclZ8v4bfds='}]
    ice_servers = [
    RTCIceServer(urls='stun:global.stun.twilio.com:3478'),  # Public STUN server
    RTCIceServer(urls='turn:global.turn.twilio.com:3478?transport=udp', 
                 credential='LfHgqhGrjZII2h3wBk4iSs/r3wl3AFe3SclZ8v4bfds=', 
                 username='78e87c2da25896f0cdf3b49542a0176afe9e9017d82a619907a2eaff9c0ed84e'),  # TURN server with UDP transport
    RTCIceServer(urls='turn:global.turn.twilio.com:3478?transport=tcp', 
                 credential='LfHgqhGrjZII2h3wBk4iSs/r3wl3AFe3SclZ8v4bfds=', 
                 username='78e87c2da25896f0cdf3b49542a0176afe9e9017d82a619907a2eaff9c0ed84e'),  # TURN server with TCP transport
    RTCIceServer(urls='turn:global.turn.twilio.com:443?transport=tcp', 
                 credential='LfHgqhGrjZII2h3wBk4iSs/r3wl3AFe3SclZ8v4bfds=', 
                 username='78e87c2da25896f0cdf3b49542a0176afe9e9017d82a619907a2eaff9c0ed84e'),  # TURN server with TCP transport on port 443
]


    configuration = RTCConfiguration(iceServers=ice_servers)

    # Then use this configuration when creating the RTCPeerConnection
    pc = RTCPeerConnection(configuration=configuration)
   

    # rtc_configuration = RTCConfiguration(iceServers=ice_servers)

    # Use rtc_configuration when creating the RTCPeerConnection
    # pc = RTCPeerConnection(rtc_configuration)
    video_track = VideoTransformTrack()
    pc.addTrack(video_track)

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            recorder.addTrack(track)
            asyncio.ensure_future(process_incoming_video(track))

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"Connection state changed to: {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()

    display_thread = None

    try:
        async with websockets.connect(SERVER_URL) as websocket:
            await websocket.send(json.dumps({"type": "join", "client_id": CLIENT_ID}))
            print(f"Joined server with client ID: {CLIENT_ID}")

            if CLIENT_ID == "client1":
                offer = await run_offer(pc)
                await websocket.send(json.dumps({"type": "offer", "offer": offer}))
                print("Sent offer")

            signaling_task = asyncio.create_task(consume_signaling(pc, websocket))
            await recorder.start()
            print("Recorder started")

            # Run the display function in a separate thread
            display_thread = asyncio.to_thread(display_video_pygame)
            
            # Wait for the signaling task and display thread to complete
            await asyncio.gather(signaling_task, display_thread)
            # if asyncio.gather(signaling_task, display_thread) is True:
            #     print("All tasks completed successfully")
            # else:
            #     remove_clients()

    except websockets.exceptions.ConnectionClosed as e:
        print(f"WebSocket connection closed unexpectedly: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        await recorder.stop()
        await pc.close()
        video_track.cap.release()
        print("Resources released and connections closed")

    

async def main_loop():
    while True:
        await start_client()

if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except Exception as e:
        print(f"Failed to start client: {e}")