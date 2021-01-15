import os
import time
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import threading
from deep.tracker.video_stream import VideoStream
from deep.tracker.deep_sort import preprocessing
from deep.tracker.deep_sort import nn_matching
from deep.tracker.deep_sort.tracker import Tracker
from deep.tracker.utils import parse_label_map
from deep.tracker.utils import initialize_detector
from deep.tracker.utils import generate_detections
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon



MODEL_NAME="/home/pi/tflite1/bir_isim/deep/detection_model/last_model.tflite"
LABEL_PATH = "/home/pi/tflite1/bir_isim/deep/detection_model/labelmap.txt"
VIDEO_PATH="/home/pi/tflite1/retail_store2.mp4"
DEFAULT_LABEL_MAP_PATH = os.path.join(os.path.dirname(__file__), LABEL_PATH)
THRESHOLD=0.5

MAX_COSINE_DIST = 0.5 #0.4
NN_BUDGET = 20
output = None
total_trackers=None

field1_dict={}
field2_dict = {}

def main():
    global output,total_trackers
    
    labels = parse_label_map(DEFAULT_LABEL_MAP_PATH)
    
    interpreter = initialize_detector(MODEL_NAME)
 
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", MAX_COSINE_DIST, NN_BUDGET)
    tracker = Tracker(metric) 
   
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    videostream = VideoStream(video_path=VIDEO_PATH).start()
    
    while True:
    
        t1 = cv2.getTickCount()
        frame = videostream.read()

        detections = generate_detections(frame, interpreter, THRESHOLD)
        

        if len(detections) == 0: print('   > no detections...')
        
        else:
            tracker.predict()
            tracker.update(detections)
            
            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
            
            total_trackers = len(tracker.tracks)
            field1_count =0
            field2_count = 0
            if len(tracker.tracks) > 0:
                for i,track in enumerate(tracker.tracks):
                    bbox = track.to_tlbr()
                    class_name = labels[track.get_class()]
                    
                    color = colors[int(track.track_id) % len(colors)]
                    color = [i * 255 for i in color]
                    
                    xmin= int(bbox[0])
                    ymin=int(bbox[1])
                    xmax=int(bbox[2])
                    ymax = int(bbox[3])
                    
                    center_x = int((xmin+xmax) / 2)
                    center_y = int((ymin+ymax) / 2)
                    
                    center_coor = (center_x,center_y)
                    cv2.circle(frame,center_coor,10,color,cv2.FILLED)
                            
                    pts_f1 = [[522,138],[1066,522],[1200,270],[580,30]]
                    pts_f2 = [[172,142],[410,607],[657,440],[319,142]]
                    
                    create_polygon(pts_f1,frame)
                    create_polygon(pts_f2,frame)
                       
                    
                    center_point = Point(center_x,center_y)
                    polygon_f1 = Polygon(pts_f1)
                    polygon_f2 = Polygon(pts_f2)
                    
                    
                    
                    
                    if is_field_contain_center(polygon_f1,center_point): # field e girmiş oluyor
                        field1_dict[track.track_id] = 1
                        field1_count = field1_count +1
                        
                                       
                        
                    if is_field_contain_center(polygon_f2,center_point): # field e girmiş oluyor
                        field2_dict[track.track_id] = 1
                        field2_count = field2_count +1
                    
                    
                    #f1_count = count_field(field1_dict,track,polygon_f1,center_point,tracker.tracks)
                    draw_bounding_box(frame,xmin,ymin,xmax,ymax,track.track_id,color)
            
            print(field1_count)
        
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)            
        frame = cv2.resize(frame,(762,432))
        
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1
        output = frame.copy()
        
 

def is_field_contain_center(polygon,center_point):
    if polygon.contains(center_point):
        return True
    else:
        return False
    
def is_id_in_dict(field_dict,track):
    track_id = track.track_id
    
    if track_id in field_dict:
        return True
    else:
        return False
    
def is_in_tracks(track,tracks):
    if track in tracks:
        return True
    else:
        return False
    
def count_field(field_dict,track,polygon,center_point,tracks):
    
    if(field_dict == False):
        return 0
    
    track_id = track.track_id
    check_result = is_id_in_dict(field_dict,track)
    
    
    # ekranda görünüp ama fiel in dışarda kalmış demek 
    if check_result == True:
        if is_field_contain_center(polygon,center_point) == False:
            field_dict.pop(track_id,None)
            print(">DELETED")
                        
    return len(field_dict)
        
    
def draw_bounding_box(frame,xmin,ymin,xmax,ymax,track_id,color):
    
    label = 'id:%s' % (str(track_id)) #(str(track.track_id))
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    label_ymin = max(int(ymin), labelSize[1] + 10)

    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin +labelSize[0],label_ymin+baseLine-10), color, cv2.FILLED)
    cv2.putText(frame, label,(xmin, int(ymin-7)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),2)
    
def create_polygon(points,frame):
    pts = np.array(points)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(frame,[pts],True,(0,0,255),3)
    
def main_thread_start():
    t = threading.Thread(target=main)
    t.daemon = True
    t.start()
