from flask import Flask, render_template, Response
import cv2
import detection
app = Flask(__name__)

camera = cv2.VideoCapture(0)  

def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  
        # read the camera frame
        
        if not success:
            break
        else:
            #returned image
            image=detection.detect(frame)
            ret, buffer = cv2.imencode('.jpg', image)
            
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace;boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('signlangui.html')


if __name__ == '__main__':
    app.run(debug=True)