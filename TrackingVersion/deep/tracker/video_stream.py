import cv2
from threading import Thread
import time

class VideoStream:
    def __init__(self,video_path=None):
        
        if  video_path != None:
            
            self.video_path = video_path
            self.stream = cv2.VideoCapture(self.video_path)
        else:
            self.video_path = None
            self.stream = cv2.VideoCapture(0)
            
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False


    def start(self):
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        while True:

            if self.stopped:
                self.stream.release()
                return
            
            
            (self.grabbed, self.frame) = self.stream.read()
            time.sleep(0.5)
            

    def read(self):
        if self.video_path !=None:
            return self.frame
        else:
            return cv2.flip(self.frame,0)

    def stop(self):
        self.stopped = True
    
   