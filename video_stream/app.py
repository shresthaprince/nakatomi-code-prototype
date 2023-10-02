from flask import Flask, Response
from flask_cors import CORS
from camera import VideoCamera

app = Flask(__name__)
CORS(app, resources={r"/video_feed": {"origins": "http://localhost:5173"}})

@app.route('/')

def gen(camera):
    while True:
        camera_frame = camera.get_frame()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + camera_frame + b'\r\n\r\n')
               
@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True, use_reloader=False)