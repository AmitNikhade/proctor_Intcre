import socketio
import pyautogui
import base64
import time
import io
from PIL import Image

sio = socketio.Client(logger=True, engineio_logger=True)

@sio.event(namespace='/screen')
def connect():
    print('Connection established to /screen namespace')

@sio.event(namespace='/screen')
def connect_error(data):
    print(f"Connection failed: {data}")

@sio.event(namespace='/screen')
def disconnect():
    print('Disconnected from /screen namespace')

def capture_and_send_screen():
    while True:
        if sio.connected:
            try:
                screenshot = pyautogui.screenshot()
                screenshot = screenshot.resize((1280, 720), Image.LANCZOS)
                buffered = io.BytesIO()
                screenshot.save(buffered, format="JPEG", quality=100)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                print(f"Sending image data. Size: {len(img_str)} bytes")
                sio.emit('screen_data', {'image': img_str}, namespace='/screen')
                time.sleep(0.1)  # Adjust as needed
            except Exception as e:
                print(f"Error capturing or sending screen: {e}")
                time.sleep(1)
        else:
            print("Not connected. Waiting...")
            time.sleep(5)

if __name__ == '__main__':
    sio.connect('https://dashboard.intellirecruit.ai', namespaces=['/screen'])
    capture_and_send_screen()
    
    