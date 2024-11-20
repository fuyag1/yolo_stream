from flask import Flask, Response
import cv2
import torch

app = Flask(__name__)

# Load the YOLOv5 model (you can specify a model size: 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def generate_frames():
    cap = cv2.VideoCapture(0)  # Open the webcam

    while True:
        success, frame = cap.read()  # Read a frame from the webcam
        if not success:
            break

        # Perform inference
        results = model(frame)

        # Render results on the frame
        frame_with_detections = results.render()[0]

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame_with_detections)
        frame = buffer.tobytes()

        # Yield the frame in a proper format for output
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
    <html>
        <head><title>Real-Time Object Detection</title></head>
        <body>
            <h1>Real-Time Object Detection Stream</h1>
            <img src="/video_feed" width="720" height="480">
        </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
