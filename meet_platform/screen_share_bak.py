# import asyncio
# import json
# import cv2
# import numpy as np
# from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
# from av import VideoFrame
# import aiohttp
# import mss
# import logging
# import traceback
# from aiortc.contrib.signaling import object_from_string, object_to_string

# logging.basicConfig(level=logging.INFO)

# class ScreenShareTrack(VideoStreamTrack):
#     def __init__(self):
#         super().__init__()
#         self.restart()

#     def restart(self):
#         self.sct = mss.mss()
#         self.monitor = self.sct.monitors[0]  # Capture the primary monitor
#         self.frame_count = 0

#     async def recv(self):
#         self.frame_count += 1
#         if self.frame_count % 30 == 0:  # Log every 30 frames
#             logging.info(f"Sending frame {self.frame_count}")
#         screenshot = self.sct.grab(self.monitor)
#         frame = np.array(screenshot)
#         frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
#         video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
#         pts, time_base = await self.next_timestamp()
#         video_frame.pts = pts
#         video_frame.time_base = time_base
#         return video_frame


# async def reset_room(signaling_url):
#     async with aiohttp.ClientSession() as session:
#         async with session.post(f"{signaling_url}/reset_room/test_room") as resp:
#             if resp.status!= 200:
#                 logging.error(f"Failed to reset room: {resp.status}")
#             #     raise Exception(f"Failed to reset room: {resp.status}")
#             # logging.info("Room reset")
            
            
# async def ensure_signaling(pc, signaling_url, max_retries=5):
#     for attempt in range(max_retries):
#         try:
#             offer = await pc.createOffer()
#             await pc.setLocalDescription(offer)
            
#             offer_dict = object_to_string(pc.localDescription)
            
#             async with aiohttp.ClientSession() as session:
#                 async with session.post(
#                     f"{signaling_url}/offer",
#                     json={"offer": offer_dict, "room": "test_room"}
#                 ) as resp:
#                     if resp.status != 200:
#                         raise Exception(f"Failed to send offer: {resp.status}")
            
#             logging.info("Offer sent to signaling server")
#             # import time
#             # time.sleep(10)
#             # if False:
#             # await reset_room(signaling_url)
                
#             while True:
#                 async with aiohttp.ClientSession() as session:
#                     async with session.get(f"{signaling_url}/get_answer/test_room") as resp:
#                         if resp.status == 200:
#                             answer_dict = await resp.json()
#                             answer = object_from_string(json.dumps(answer_dict))
#                             await pc.setRemoteDescription(answer)
#                             logging.info("Answer received and set")
#                             return
#                         elif resp.status != 404:
#                             raise Exception(f"Unexpected status when getting answer: {resp.status}")
#                 await asyncio.sleep(1)
#         except Exception as e:
#             logging.error(f"Signaling attempt {attempt + 1} failed: {e}")
#             if attempt == max_retries - 1:
#                 raise
#         await asyncio.sleep(2 ** attempt)  # Exponential backoff

# async def reset_peer_connection(pc):
#     if pc:
#         await pc.close()
#     new_pc = RTCPeerConnection()
#     new_pc.addTrack(ScreenShareTrack())
    
#     @new_pc.on("connectionstatechange")
#     async def on_connectionstatechange():
#         logging.info(f"Connection state is {new_pc.connectionState}")
#         if new_pc.connectionState == "connected":
#             logging.info("Peer connection established!")
#         elif new_pc.connectionState in ["failed", "disconnected", "closed"]:
#             logging.warning(f"Connection state changed to {new_pc.connectionState}")

#     return new_pc

# async def check_stop_signal(signaling_url):
#     async with aiohttp.ClientSession() as session:
#         async with session.get(f"{signaling_url}/get_stop_signal") as resp:
#             if resp.status == 200:
#                 stop_signal = await resp.json()
#                 if stop_signal.get('stop', False):
#                     logging.info("Stop signal received. Shutting down.")
#                     return True
#     return False

# async def heartbeat(pc, reset_func, signaling_url):
#     while True:
#         if pc.connectionState in ["failed", "disconnected", "closed"]:
#             logging.info("Connection lost. Attempting to reconnect...")
#             return  # Exit heartbeat, main loop will handle reconnection

#         # Check for stop signal
#         if await check_stop_signal(signaling_url):
#             break

#         await asyncio.sleep(5)  

# async def restart_ice(pc):
#     logging.info("Restarting ICE")
#     offer = await pc.createOffer()
#     await pc.setLocalDescription(offer)
#     # The answer will be set in the ensure_signaling function




# async def main():
#     signaling_url = "https://dash.intellirecruit.ai"  # Update this to your signaling server URL
#     room = "test_room"
#     pc = None

#     while True:
#         try:
#             await reset_room(signaling_url)
#             pc = await reset_peer_connection(pc)
#             await restart_ice(pc)
#             await ensure_signaling(pc, signaling_url)
#             logging.info("Screen sharing started")
            
#             heartbeat_task = asyncio.create_task(heartbeat(pc, reset_peer_connection, signaling_url, room))
#             await heartbeat_task
            
#             # If the heartbeat task ends (due to stop signal), exit the loop
#             break
#         except Exception as e:
#             logging.error(f"An error occurred: {e}")
#             logging.error(traceback.format_exc())
#             await asyncio.sleep(5)  # Wait before retrying

# if __name__ == "__main__":
#     try:
#         asyncio.run(main())
#     except Exception as e:
#         print("An error occurred:", e)






import asyncio
import json
import cv2
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
import aiohttp
import mss
import logging
import traceback
import sys
from aiortc.contrib.signaling import object_from_string, object_to_string

logging.basicConfig(level=logging.INFO)

class ScreenShareTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.restart()

    def restart(self):
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[0]  # Capture the primary monitor
        self.frame_count = 0

    async def recv(self):
        self.frame_count += 1
        if self.frame_count % 30 == 0:  # Log every 30 frames
            logging.info(f"Sending frame {self.frame_count}")
        screenshot = self.sct.grab(self.monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        pts, time_base = await self.next_timestamp()
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame

async def reset_room(signaling_url):
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{signaling_url}/reset_room/test_room") as resp:
            if resp.status != 200:
                logging.error(f"Failed to reset room: {resp.status}")
                
                
async def check_stop_signal(signaling_url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{signaling_url}/get_stop_signal/test_room") as resp:
                if resp.status == 200:
                    stop_signal = await resp.json()
                    
                    return stop_signal.get('stop', False)
    except Exception as e:
        logging.error(f"Error checking stop signal: {e}")
    return False

async def ensure_signaling(pc, signaling_url, max_retries=5):
    for attempt in range(max_retries):
        try:
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)
            
            offer_dict = object_to_string(pc.localDescription)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{signaling_url}/offer",
                    json={"offer": offer_dict, "room": "test_room"}
                ) as resp:
                    if resp.status != 200:
                        raise Exception(f"Failed to send offer: {resp.status}")
            
            logging.info("Offer sent to signaling server")
            
            while True:
                if await check_stop_signal(signaling_url):
                    logging.info("Received stop signal from server")
                    
                    await reset_room(signaling_url)
                    await reset_peer_connection(pc)
                    await main()
                    # await pc.close()
                    # return "STOP"

                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{signaling_url}/get_answer/test_room") as resp:
                        if resp.status == 200:
                            answer_dict = await resp.json()
                            answer = object_from_string(json.dumps(answer_dict))
                            await pc.setRemoteDescription(answer)
                            logging.info("Answer received and set")
                            return "CONNECTED"
                        elif resp.status != 404:
                            raise Exception(f"Unexpected status when getting answer: {resp.status}")
                await asyncio.sleep(1)
        except Exception as e:
            logging.error(f"Signaling attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
        await asyncio.sleep(2 ** attempt)  # Exponential backoff

async def reset_peer_connection(pc):
    if pc:
        await pc.close()
    new_pc = RTCPeerConnection()
    new_pc.addTrack(ScreenShareTrack())
    
    @new_pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logging.info(f"Connection state is {new_pc.connectionState}")
        if new_pc.connectionState == "connected":
            logging.info("Peer connection established!")
        elif new_pc.connectionState in ["failed", "disconnected", "closed"]:
            logging.warning(f"Connection state changed to {new_pc.connectionState}")

    return new_pc

async def heartbeat(pc, reset_func):
    while True:
        if pc.connectionState in ["failed", "disconnected", "closed"]:
            logging.info("Connection lost. Attempting to reconnect...")
            return  # Exit heartbeat, main loop will handle reconnection
        await asyncio.sleep(5)  # Check every 5 seconds

async def restart_ice(pc):
    logging.info("Restarting ICE")
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

async def main():
    signaling_url = "https://dash.intellirecruit.ai"  # Update this to your signaling server URL
    pc = None

    while True:
        try:
            await reset_room(signaling_url)
            pc = await reset_peer_connection(pc)
            await restart_ice(pc)
            result = await ensure_signaling(pc, signaling_url)
            if result == "STOP":
                logging.info("Stopping the client as requested by the server")
                break
            logging.info("Screen sharing started")
            
            while True:
                if await check_stop_signal(signaling_url):
                    
                    logging.info("Stopping the client as requested by the server")
                    await reset_room
                    if pc:
                        await pc.close()
                    pc = await reset_peer_connection(None)
                    await pc.close()
                    return
                await asyncio.sleep(5)  # Check every 5 seconds
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            logging.error(traceback.format_exc())
            await asyncio.sleep(5)  # Wait before retrying
            
if __name__ == "__main__":
    try:
        # import requests
        # response = requests.post('https://dash.intellirecruit.ai/stop_client/test_room')
        # asyncio.sleep(10) # Wait before retrying
        asyncio.run(main())
    except Exception as e:
        print("An error occurred:::::::::::::::::::::::::::::", e)




# import requests
# response = requests.post('https://dash.intellirecruit.ai/stop_client/test_room')