import cv2
import numpy as np
import threading
from flask import Flask,render_template,Response
from flask_socketio import SocketIO,emit
from tflite_runtime.interpreter import Interpreter
from datetime import datetime
import deep.tracker.main as Main


app = Flask(__name__)
app.config['SECRET_KEY'] ='SECRET!'
socketio=SocketIO(app,async_mode='eventlet')


generated_time=None
output = None
total_trackers=None


@socketio.on('my event')
def handle_connect():
    my_json = {'total_trackers':str(total_trackers),
               "time":str(datetime.now().strftime("%Y.%m.%d - %H:%M:%S"))}
    emit('my response',my_json)
    socketio.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

       
def get_output():
    global output,total_trackers
    while True:
        output = Main.output
        total_trackers = Main.total_trackers

        
def generate():
    global output
    while True:
        if output is None:
            break
        (flag,encoded) = cv2.imencode(".jpg",output)
        
        if not flag:
            break

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encoded) + b'\r\n')
     
               
@app.route("/video_feed")
def video_feed():
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")



if __name__ == '__main__':

    Main.main_thread_start()

    t = threading.Thread(target=get_output)
    t.daemon = True
    t.start()
    socketio.run(app,host="192.168.1.38",port="5000",debug=True)


 

  
   
  
