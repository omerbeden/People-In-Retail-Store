import os
from time import sleep
import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np
import cv2
from scipy.spatial.distance import cosine
from deep.tracker.deep_sort import generate_detections as gd
from deep.tracker.deep_sort.detection import Detection
from deep.tracker.deep_sort.preprocessing import non_max_suppression
nms_max_overlap = 1.0


w_path = os.path.join(os.path.dirname(__file__), 'deep_sort/mars-small128.pb')
encoder = gd.create_box_encoder(w_path, batch_size=1)
    

def initialize_detector(model_name):
    model_path = os.path.join(os.path.dirname(__file__), model_name)    
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    return interpreter


def generate_detections(cv2_image, interpreter, threshold):
    
    height = interpreter.get_input_details()[0]['shape'][1]
    width = interpreter.get_input_details()[0]['shape'][2]
    
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

    bboxes = np.squeeze(interpreter.get_tensor(interpreter.get_output_details()[0]['index']))
    classes = np.squeeze(interpreter.get_tensor(interpreter.get_output_details()[1]['index'])) 
    scores = np.squeeze(interpreter.get_tensor(interpreter.get_output_details()[2]['index']))

    keep_idx = np.less(scores[np.greater(scores, threshold)], 1)
    bboxes  = bboxes[:keep_idx.shape[0]][keep_idx]
    classes = classes[:keep_idx.shape[0]][keep_idx]
    scores = scores[:keep_idx.shape[0]][keep_idx]
    
    if len(keep_idx) > 0:
        
        bboxes[:,0] = bboxes[:,0] * cv2_image.shape[0]
        bboxes[:,1] = bboxes[:,1] * cv2_image.shape[1]
        bboxes[:,2] = bboxes[:,2] * cv2_image.shape[0]
        bboxes[:,3] = bboxes[:,3] * cv2_image.shape[1]
    
    for box in bboxes:
        xmin = int(box[1])
        ymin = int(box[0])
        w = int(box[3]) - xmin
        h = int(box[2]) - ymin
        box[0], box[1], box[2], box[3] = xmin, ymin, w, h
        

    features = encoder(cv2_image, bboxes)

    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, classes, features)]

    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    indices = non_max_suppression(boxes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]
    
    return detections


def parse_label_map(label_map_path):
    labels = {}
    for i, row in enumerate(open(label_map_path)):
        labels[i] = row.replace('\n','')
    return labels
