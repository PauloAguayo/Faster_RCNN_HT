import cv2
import numpy as np
import os

class Record_video(object):
    def __init__(self,name_video,size,ask_chart):
        self.ask_chart = ask_chart
        if self.ask_chart:
            if size[0]>size[1]:
                self.ax = 0
                self.resize = (1080,640)
                self.resize_writer = (1080,640*2)
            elif size[0]<size[1]:
                self.ax = 1
                self.resize = (720,900)
                self.resize_writer = (720*2,900)
            else:
                self.ax = 1
                self.resize = (720,720)
                self.resize_writer = (720*2,720)
        else:
            self.resize = size
            self.resize_writer = self.resize

        self.fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.out = cv2.VideoWriter(name_video,self.fourcc, 20.0, (self.resize_writer))
        self.picture = "chart.png"

    def Recording(self,frame):
        if self.ask_chart:
            self.frame = cv2.resize(frame, self.resize, interpolation = cv2.INTER_AREA)
            self.chart = cv2.imread(self.picture)
            self.chart = cv2.resize(self.chart, self.resize, interpolation = cv2.INTER_AREA)
            self.both = np.concatenate((self.frame,self.chart),axis=self.ax)
            self.out.write(self.both)
        else:
            self.out.write(frame)

    def End_recording(self):
        self.out.release()
        os.remove(self.picture)
