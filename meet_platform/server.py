import asyncio
import json
from aiohttp import web
import aiohttp_cors
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

clients = {}
pending_offers = {}

async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    client_id = None

    async for msg in ws:
        if msg.type == web.WSMsgType.TEXT:
            data = json.loads(msg.data)
            logger.info(f"Received message: {data}")
            if data["type"] == "join":
                client_id = data["client_id"]
                clients[client_id] = ws
                logger.info(f"Client {client_id} joined")

                # Send pending offer if exists
                other_client = "client2" if client_id == "client1" else "client1"
                if other_client in pending_offers:
                    await ws.send_json(pending_offers[other_client])
                    logger.info(f"Sent pending offer to {client_id}")
                    del pending_offers[other_client]

            elif data["type"] in ["offer", "answer"]:
                other_client = "client2" if client_id == "client1" else "client1"
                if other_client in clients:
                    await clients[other_client].send_json(data)
                    logger.info(f"Sent {data['type']} to {other_client}")
                else:
                    pending_offers[client_id] = data
                    logger.warning(f"No client {other_client} found to send {data['type']}, storing offer")

    if client_id:
        del clients[client_id]
        logger.info(f"Client {client_id} disconnected")

    return ws

app = web.Application()
cors = aiohttp_cors.setup(app)

resource = cors.add(app.router.add_resource("/websocket"), {
    "*": aiohttp_cors.ResourceOptions(
        allow_credentials=True,
        expose_headers="*",
        allow_headers="*"
    )
})
resource.add_route("GET", websocket_handler)

web.run_app(app, port=1234)
