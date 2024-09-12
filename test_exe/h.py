import asyncio
import logging
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
import aiohttp

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class DummyAudioTrack(MediaStreamTrack):
    kind = "audio"

    async def recv(self):
        await asyncio.sleep(0.02)
        return None

class SimpleClient:
    def __init__(self, server_url='https://dash.intellirecruit.ai'):
        self.server_url = server_url
        self.pc = RTCPeerConnection()
        
        # Add a dummy audio track
        self.pc.addTrack(DummyAudioTrack())

    async def connect(self):
        # create offer
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)

        # send offer and get answer
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(f"{self.server_url}/offer", 
                                        json={"sdp": self.pc.localDescription.sdp, 
                                              "type": self.pc.localDescription.type}) as resp:
                    if resp.status == 200:
                        answer = await resp.json()
                        await self.pc.setRemoteDescription(
                            RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))
                        logger.info("Connected to server")
                    else:
                        logger.error(f"Server returned status {resp.status}: {await resp.text()}")
            except aiohttp.ClientError as e:
                logger.error(f"Error connecting to server: {e}")
                raise

    async def run(self):
        try:
            await self.connect()
            # Keep the connection alive
            while True:
                await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Error in run: {e}")
        finally:
            await self.pc.close()

async def main():
    client = SimpleClient()
    await client.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Client stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)