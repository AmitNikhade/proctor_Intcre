# from flask import Flask, request, jsonify, Response
# import monitor
# import numpy as np
# import cv2

# app = Flask(__name__)
# print("Starting server")

# @app.route('/post-data', methods=['POST'])
# def post_data():
#     # Read the image file sent in the 'image' field
#     file = request.files['image'].read()
    
#     # Convert the binary data to a numpy array
#     npimg = np.frombuffer(file, np.uint8)
    
#     # Decode the image
#     frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

#     if frame is None:
#         return jsonify({'message': 'Error in decoding image'}), 400

#     # Process the frame
#     processed_frame = monitor.process_frame(frame)

#     # Encode the processed frame as JPEG before sending
#     ret, jpeg = cv2.imencode('.jpg', processed_frame)
    
#     if not ret:
#         return jsonify({'message': 'Error in encoding frame'}), 500

#     # Send the processed image as a binary response
#     return Response(jpeg.tobytes(), mimetype='image/jpeg')

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0')
print("1")
from flask import Flask, request, jsonify, Response
print("1")
import monitor
print("1")
import numpy as np
print("1")
import cv2
print("1")
import logging
print("1")
import flask
print(flask.__version__)

# Set up logging to output to the console
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
print("Starting server")
logging.info("Starting server")

@app.route('/post-data', methods=['POST'])
def post_data():
    print("Received a POST request")
    logging.info("Received a POST request")

    # Read the image file sent in the 'image' field
    file = request.files['image'].read()
    print("Image file read successfully")
    logging.info("Image file read successfully")

    # Convert the binary data to a numpy array
    npimg = np.frombuffer(file, np.uint8)
    print("Converted image file to numpy array")
    logging.info("Converted image file to numpy array")

    # Decode the image
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    print("Image decoded")
    logging.info("Image decoded")

    if frame is None:
        print("Error: Decoded image is None")
        logging.error("Error: Decoded image is None")
        return jsonify({'message': 'Error in decoding image'}), 400

    # Process the frame
    print("Processing the image frame")
    logging.info("Processing the image frame")
    processed_frame = monitor.process_frame(frame)
    print("Image frame processed")
    logging.info("Image frame processed")

    # Encode the processed frame as JPEG before sending
    ret, jpeg = cv2.imencode('.jpg', processed_frame)
    print("Processed frame encoded to JPEG")
    logging.info("Processed frame encoded to JPEG")

    if not ret:
        print("Error: Encoding frame to JPEG failed")
        logging.error("Error: Encoding frame to JPEG failed")
        return jsonify({'message': 'Error in encoding frame'}), 500

    # Send the processed image as a binary response
    print("Sending the processed image as response")
    logging.info("Sending the processed image as response")
    return Response(jpeg.tobytes(), mimetype='image/jpeg')

if __name__ == '__main__':
    print("Running Flask server")
    logging.info("Running Flask server")
    app.run(debug=True, host='0.0.0.0', port = 8080)
