
import os
import argparse
import cv2
import numpy as np
import sys
import time
import threading
from datetime import datetime
from flask import Flask,render_template,Response,send_from_directory,abort,send_file
from flask_socketio import SocketIO,emit
from tflite_runtime.interpreter import Interpreter
from utils import generate_detections,initialize_detector
from utils import read_labels,draw_bounding_boxes,create_polygon,is_field_contain_center,firebase_post,firebase_get,make_report
from video_stream import VideoStream

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


MODEL_NAME = "detection_model/last_model.tflite"
LABELMAP_NAME = "detection_model/labelmap.txt"
VIDEO_PATH = "retail_store2.mp4"

CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME)
PATH_TO_LABELS = os.path.join(CWD_PATH,LABELMAP_NAME)

THRESHOLD=0.5
imW, imH = 1270,720
output = None
heatmap_image = None
generated_time=None
total_people=0
field1_count =0
field2_count = 0

app = Flask(__name__)
app.config['SECRET_KEY'] ='SECRET!'
socketio=SocketIO(app)
report_file_name = "myreport.xlsx"


@socketio.on("report_event")
def handle_report_event():
    firebase_result = firebase_get()
    response = make_report(firebase_result,report_file_name)
    emit("report response",str(response))
    socketio.sleep(0.2)


@socketio.on("firebase_post_event")
def handle_firebase_event():
    json = {"date":str(datetime.now().strftime("%d.%m.%Y - %H:%M:%S")),
            "total people":str(total_people),
            "field1 count":str(field1_count),
            "field2 count":str(field2_count)}
    if firebase_post(json):
        print("posted to firebase")
    else:
        print("post not work!!")
    socketio.sleep(0.5)

@socketio.on('update_all_data_event')
def update_all_data_event():
    my_json = {"total_people":str(total_people),
               "field1_count":str(field1_count),
               "field2_count":str(field2_count),
               "time":str(datetime.now().strftime("%Y.%m.%d - %H:%M:%S"))}
    emit('my response',my_json)
    socketio.sleep(0.1)
 
@app.route('/download_report')
def download_report():
    try:
        return send_file("/home/pi/tflite1/bir_isim/no_tracking/report/"+report_file_name)
    except FileNotFoundError:
        abort(404)
        
@app.route('/')
def index():
    return render_template('index.html')


labels = read_labels(PATH_TO_LABELS)
interpreter = initialize_detector(MODEL_NAME)

def m():
    global output,heatmap_image,total_people,field1_count,field2_count
    
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    videostream = VideoStream(VIDEO_PATH).start()
    
    color_f1 = (0,0,255)
    color_f2 = (255,0,0)
    heatmap = np.zeros((720,1270,3),dtype=np.uint8)
    ht_color = (191,255,1)
    
    while True:
        t1 = cv2.getTickCount()
        
        frame1 = videostream.read()
        frame = frame1.copy()
        boxes,classes,scores = generate_detections(frame,interpreter)
        
        total_people=0
        field1_count =0
        field2_count = 0
        for i in range(len(scores)):
            if ((scores[i] > THRESHOLD) and (scores[i] <= 1.0)):
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                total_people = total_people +1
               
                center_x = int((xmin+xmax) / 2)
                center_y = int((ymin+ymax) / 2)
                center_coor = (center_x,center_y)
                
                color_bb = (10,200,10)
                cv2.circle(frame,center_coor,10,color_bb,cv2.FILLED)
                
                pts_f1 = [[522,138],[1066,522],[1200,270],[580,30]]
                pts_f2 = [[172,142],[410,607],[657,440],[319,142]]
                            
                create_polygon(pts_f1,frame,color_f1)
                create_polygon(pts_f2,frame,color_f2)
                   
                center_point = Point(center_x,center_y)
                polygon_f1 = Polygon(pts_f1)
                polygon_f2 = Polygon(pts_f2)
                
                if is_field_contain_center(polygon_f1,center_point): 
                    field1_count = field1_count +1
                    color_bb = color_f1
                    
                if is_field_contain_center(polygon_f2,center_point): 
                    field2_count = field2_count +1
                    color_bb = color_f2
                
                draw_bounding_boxes(frame,classes,xmin,xmax,ymin,ymax,color_bb,labels)
                
                if (heatmap[center_y,center_x][0] != 0) and (heatmap[center_y,center_x][1] != 0) and (heatmap[center_y,center_x][2] != 0):
                    b = heatmap[center_y,center_x][0]
                    g = heatmap[center_y,center_x][1]
                    r = heatmap[center_y,center_x][2]
                    
                    b= b - b*0.5
                    g= g - g*0.2
                    r= r + r*0.5
                    
                    cv2.circle(heatmap,center_coor,10,(b,g,r),cv2.FILLED)
                else:
                    cv2.circle(heatmap,center_coor,10,ht_color,cv2.FILLED)
                    

        
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
        frame = cv2.resize(frame,(698,396))
        
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1
   
        overlay = frame1
        alpha_backgroud = 0.7
        alpha_heatmap = 0.9
        cv2.addWeighted(overlay,alpha_heatmap,frame1,1-alpha_heatmap,0,frame1)
        cv2.addWeighted(heatmap,alpha_backgroud,frame1,1-alpha_backgroud,0,frame1)
        
        frame2 = cv2.resize(frame1,(698,396))
        
        output = frame.copy()
        heatmap_image = frame2

    
    
def generate_video_feed():
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
    return Response(generate_video_feed(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")


@app.route("/heatmap_feed")
def heatmap_feed():
    def generate_heatmap():
        global heatmap_image
        while True:
            if heatmap_image is None:
                break
            (flag,encoded) = cv2.imencode(".jpg",heatmap_image)
            if not flag:
                break

            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                bytearray(encoded) + b'\r\n')
    
    return Response(generate_heatmap(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
  
  t = threading.Thread(target=m)
  t.daemon = True  
  t.start()

  socketio.run(app,host="192.168.1.38",port="5000",debug=True)
