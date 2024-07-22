

import socketio
import pyautogui
import io
import base64
import time

from PIL import Image


sio = socketio.Client(logger=True, engineio_logger=True)



def capture_and_send_screen():
    sio.connect('https://dashboard.intellirecruit.ai', namespaces=['/screen'])
    while True:
        if sio.connected:
            try:
                screenshot = pyautogui.screenshot()
                screenshot = screenshot.resize((640, 360), Image.LANCZOS)
                buffered = io.BytesIO()
                screenshot.save(buffered, format="JPEG", quality=85)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                print(f"Sending image data. Size: {len(img_str)} bytes")
                sio.emit('screen_data', {'image': img_str}, namespace='/screen')
                # time.sleep(0.1)  # Adjust as needed
            except Exception as e:
                print(f"Error capturing or sending screen: {e}")
                # time.sleep(1)
        else:
            print("Not connected. Waiting...")
            time.sleep(5)
        time.sleep(0.01)
        
        
        
capture_and_send_screen()