from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
from fer import FER
import os

# Initialize Flask app
app = Flask(__name__)

# Load and optimize the emotion detection model (use a lighter model or apply quantization if possible)
detector = FER(mtcnn=True)

# Global variable to store the video path
video_path = None


# Function to process video frames
def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video stream or file")
        return None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for faster processing
        frame = cv2.resize(frame, (480, 360))

        # Perform emotion detection on every 2nd frame to speed up processing
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 2 == 0:
            result = detector.detect_emotions(frame)

            for face in result:
                bounding_box = face["box"]
                emotions = face["emotions"]
                max_emotion = max(emotions, key=emotions.get)

                x, y, w, h = bounding_box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, max_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index.html', video=video_path)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/', methods=['POST'])
def upload_video():
    global video_path
    file = request.files['file']
    if file and file.filename.endswith(('mp4', 'avi', 'mov')):
        video_path = os.path.join('static/uploads', file.filename)
        file.save(video_path)
        return redirect(url_for('index'))
    return "Invalid file format. Please upload a video file."


if __name__ == '__main__':
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')
    app.run(debug=True)
