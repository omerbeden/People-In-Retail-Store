import os
from time import sleep
import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
from firebase import firebase
import pandas as pd
import json

CWD_PATH = os.getcwd()
firebase_url = "firebase-url"
firebase = firebase.FirebaseApplication(firebase_url,None)
db_id=0

def make_report(data,filename):
    data_frame = pd.DataFrame(data)
    data_frame = data_frame.astype({"field1 count":int,"field2 count":int, "total people":int})
    writer = pd.ExcelWriter(os.path.join(CWD_PATH,filename),engine="xlsxwriter")
    start_row= 2
    data_frame.to_excel(writer,sheet_name="Sheet1",startrow=start_row)
    
    book = writer.book
    sheet = writer.sheets["Sheet1"]
    
    bold = book.add_format({'bold':True,'size':24})
    sheet.write('A1','My Report',bold)
    row_count = data_frame.shape[0] + start_row +2

    chart_line = book.add_chart({'type':'line'})

    chart_line.add_series({'name': '=Sheet1!$E$3',
                      'categories': '=Sheet1!$A$4:$A${}'.format(str(row_count)),
                      'values': '=Sheet1!$E$4:$E${}'.format(str(row_count)),
        })
    chart_line.set_style(10)
    sheet.insert_chart('G2',chart_line)
    
    chart_col = book.add_chart({'type':'column'})
    chart_col.add_series({ 'name': '=Sheet1!$C$3',
                           'categories': '=Sheet1!$A$4:$A${}'.format(str(row_count)),
                           'values': '=Sheet1!$C$4:$C${}'.format(str(row_count)),
                    
        })
    chart_col.add_series({ 'name': '=Sheet1!$D$3',
                           'values': '=Sheet1!$D$4:$D${}'.format(str(row_count)),
        })
    
    chart_col.set_title({'name':'Field1 and Field2'})
    chart_col.set_x_axis({'name':'Date id'})
    chart_col.set_y_axis({'name':'Count'})
    
    sheet.insert_chart('P2',chart_col)
    format1 = book.add_format({'font_color':'#E93423'})
    
    writer.save()
    if not data_frame.empty:
        print("report created!")
        return True

    else:
        return False


def firebase_get():
    result = firebase.get(firebase_url,None)
    return result
    
def firebase_post(data):
    global db_id
    result = firebase.patch(firebase_url+"/"+str(db_id),data)
    db_id = db_id +1
    
    if result != None:
        return True
    elif result == None:
        return False


def is_field_contain_center(polygon,center_point):
    if polygon.contains(center_point):
        return True
    else:
        return False
    
def create_polygon(points,frame,color):
    pts = np.array(points)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(frame,[pts],True,color,3)
    
def read_labels(PATH_TO_LABELS):
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

def initialize_detector(model_name):
    model_path = os.path.join(os.path.dirname(__file__), model_name)    
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    return interpreter


def draw_bounding_boxes(frame,classes,xmin,xmax,ymin,ymax,color,labels):
    object_name = labels[int(classes[0])] 
    label = '%s' % (object_name)
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) 
    label_ymin = max(ymin, labelSize[1] + 10) 
    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)
    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
def generate_detections(cv2_image, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    
    frame_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)
    
    val = np.reshape(frame_resized[:,:,0],-1)
    input_mean = 127.5 #np.mean(val)
    input_std = 127.5 #np.std(val)
    
    floating_model = (interpreter.get_input_details()[0]['dtype'] == np.float32)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std
        
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0] 
    classes = interpreter.get_tensor(output_details[1]['index'])[0] 
    scores = interpreter.get_tensor(output_details[2]['index'])[0] 

    return boxes,classes,scores

