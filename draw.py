import cv2
import numpy as np
from math import hypot

class Drawing(object):
    def __init__(self,shape):
        self.shape = shape  # x,y
        self.max_length = np.maximum(self.shape[0],self.shape[1])
        self.min_length = np.minimum(self.shape[0],self.shape[1])
        self.image_center = np.array([int(self.shape[0]/2),int(self.shape[1])]) # x,y
        #self.image_center = np.array([int(self.shape[0]/2),int(self.shape[1]/2)]) # x,y
#        self.image_center = np.array([int(self.shape[0]/2),int(self.shape[1]*0.8)]) # x,y
        #self.image_center = np.array([int(self.shape[0]/2),int(self.shape[1]*0.35)]) # x,y

    def Draw_detections(self,frame,centroid_box,length_box,H=2.5,h=1.7):
        #gamma = np.linalg.norm(polygon_centroid-centroid_box)
        gamma = np.linalg.norm(centroid_box-self.image_center) # pixels distance
        # pixels distance
        #gamma_y_pixels = abs(self.image_center[1]-centroid_box[1])
        #gamma_x_pixels = abs(self.image_center[0]-centroid_box[0])
                #gamma *= ((np.pi/4)/(self.min_length/2))
#        gamma *= ((np.pi/4)/(self.max_length/2))
        gamma *= ((np.pi/4)/(self.max_length))
        # from pixels to radians
        #gamma_y = gamma_y_pixels*((np.pi/4)/(self.min_length/2))
        #gamma_x = gamma_x_pixels*((np.pi/3)/(self.max_length/2))
        #gamma = hypot(gamma_x,gamma_y)
        d1_prima = np.tan(gamma)*H
        d1 = d1_prima - np.tan(gamma)*h
        alpha = np.arctan(d1/H)
        beta = abs(gamma) - abs(alpha)
        slope = (centroid_box[1]-self.image_center[1])/(centroid_box[0]-self.image_center[0])
        if abs(slope)>=1:
            distance_box = length_box[0]
            d = 1
        elif abs(slope)<1:
            distance_box = length_box[1]
            d = 0
        x_param = 2*centroid_box[0]/self.shape[0]-1
        y_param = 2*centroid_box[1]/self.shape[1]-1
        #y_param = centroid_box[1]/self.image_center[1]-1
        y_axis = []

        if x_param>=0 and y_param>=0: # 4th
            if d==0:
                x_axis = np.arange(self.image_center[0],centroid_box[0]+1)
                if slope >1: slope = abs(slope) - abs(int(slope))
            else:
                x_axis = np.arange(self.image_center[1],centroid_box[1]+1)
                if slope <1: slope+=1
            x_axis = x_axis[::-1]
        elif x_param<0 and y_param>=0:  # 3rd
            if d==0:
                x_axis = np.arange(centroid_box[0],self.image_center[0]+1)                            #    2  |  1
                if abs(slope) >1: slope = (-1)*(abs(slope) - abs(int(slope)))                         #  _____|_____
            else:                                                                                     #   3   |  4      quadrants
                x_axis = np.arange(self.image_center[1],centroid_box[1]+1)                            #       |
                x_axis = x_axis[::-1]
                if abs(slope)<1: slope-=1
        elif x_param>=0 and y_param<0:   # 1st
            if d==0:
                x_axis = np.arange(self.image_center[0],centroid_box[0]+1)
                if abs(slope) >1: slope = (-1)*(abs(slope) - abs(int(slope)))
                x_axis = x_axis[::-1]
            else:
                x_axis = np.arange(centroid_box[1],self.image_center[1]+1)
                if abs(slope) <1: slope-=1
        elif x_param<0 and y_param<0:    # 2nd
            if d==0:
                x_axis = np.arange(centroid_box[0],self.image_center[0]+1)
                if slope >1: slope = slope - int(slope)
            else:
                x_axis = np.arange(centroid_box[1],self.image_center[1]+1)
                if slope <1: slope+=1

        if d == 0:
            def straight(x__,x,y,slope):
                return(slope*(x__-x)+y)
        else:
            def straight(x__,x,y,slope): # do not get confused!!! this x__ is a y__
                return((x__-y)/float(slope)+x)

        x_axis = np.array(x_axis)
        middle = int(distance_box/2)
        for x__ in x_axis:
            y_temp = []
            y__ = straight(x__,centroid_box[0],centroid_box[1],slope)
            y_plus = y__
            y_minus = y__
            for q in np.arange(1,1+distance_box):
                if q > middle:
                    y_plus +=1
                    y_temp.append(int(y_plus))
                elif q <= middle:
                    y_minus -=1
                    y_temp.append(int(y_minus))
            y_axis.append(y_temp)
        y_axis = np.array(y_axis)

        # optimization points
        lim = 0
        #if x_param<0 and y_param<0:
            #beta_radians = abs(np.arctan((centroid_box[1]-self.image_center[1])/(centroid_box[0]-self.image_center[0]))-np.pi/2)
        #    beta_radians = np.arctan((centroid_box[1]-self.image_center[1])/(centroid_box[0]-self.image_center[0]))
        #    print(beta_radians)
        #    beta_x_pixels = beta*np.sin(beta_radians)
        #    beta_y_pixels = beta*np.cos(beta_radians)
        #else:
        #    beta_radians = abs(np.arctan((centroid_box[1]-self.image_center[1])/(centroid_box[0]-self.image_center[0])))-np.pi/2
        #    beta_x_pixels = beta*np.cos(beta_radians)
        #    beta_y_pixels = beta*np.sin(beta_radians)
        #beta_x = beta_x_pixels*((self.max_length/2)/(np.pi/3))
        #beta_y = beta_y_pixels*((self.min_length/2)/(np.pi/4))
        #beta = hypot(beta_x,beta_y)
#        beta*=((self.max_length/2)/(np.pi/4))
        beta*=((self.max_length)/(np.pi/4))
        for x_p,y_row in zip(x_axis,y_axis):
            for y_p in y_row:
                if d == 0:
                    center2Coord = float(np.linalg.norm(np.array([x_p,y_p])-self.image_center))
                    #cv2.circle(frame,(x_p,y_p),1,(255,255,0),-1)
                else:
                    center2Coord = float(np.linalg.norm(np.array([y_p,x_p])-self.image_center))
                    #cv2.circle(frame,(y_p,x_p),1,(255,255,0),-1)
                if float(center2Coord)<=abs(beta):
                    if center2Coord>int(lim):
                        lim = center2Coord
                        if d == 0:
                            coords = (y_p,x_p)
                        else:
                            coords = (x_p,y_p)
        try:
            #if np.linalg.norm(np.array([coords[1],coords[0]])-centroid_box)>250:
            #    return((0,0))
            #else:
            #    return(coords)
            return(coords)
        except:
            return((0,0))


    def Generate_Polygon(self, nameWindow,frame):
        def draw_circle(event,x,y,flags,param):
            global mouseX,mouseY
            if event == cv2.EVENT_LBUTTONDBLCLK:
                cv2.circle(frame,(x,y),6,(0,255,255),-1)
                mouseX,mouseY = x,y

        cv2.namedWindow(nameWindow)
        cv2.setMouseCallback(nameWindow,draw_circle)
        points = []
        point_counter = 0
        while(1):
            cv2.imshow(nameWindow,frame)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
            elif k == ord('a'):
                point_counter+=1
                points.append([mouseX,mouseY])
                cv2.circle(frame,(mouseX,mouseY),8,(0,0,0),3)
                cv2.putText(frame, str(point_counter), (mouseX,mouseY-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
            elif k == ord('r'):
                for i,c in zip(points,np.arange(1,point_counter+1)):
                    cv2.circle(frame,(i[0],i[1]),8,(255,255,255),-1)
                    cv2.line(frame,(i[0]-6,i[1]-6),(i[0]+6,i[1]+6),(0,0,0),2)
                    cv2.line(frame,(i[0]+6,i[1]-6),(i[0]-6,i[1]+6),(0,0,0),2)
                    cv2.putText(frame, str(c), (i[0],i[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
                points = []
                point_counter = 0
        cv2.destroyAllWindows()
        return(points)

    def Square_meter(self,frame):
        self.drawing = False # True if mouse is pressed
        self.ix,self.iy = -1,-1

        def draw_square(event,x,y,flags,param):
            global xx,yy
            self.origin = np.array([0,0])
            if x<0: x = 0
            if y<0: y = 0

            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.ix,self.iy = x,y
                cv2.line(frame,(self.ix-30,self.iy),(self.ix+30,self.iy),(0,0,0),2)
                cv2.line(frame,(self.ix,self.iy-30),(self.ix,self.iy+30),(0,0,0),2)
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.drawing == True:
                    dif_x = x-self.ix
                    dif_y = y-self.iy
                    cv2.line(frame,(self.ix,self.iy),(self.ix+dif_x,self.iy),(0,0,0),2)
                    cv2.line(frame,(self.ix,self.iy),(self.ix,self.iy+dif_y),(0,0,0),2)
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                xx,yy = x,y
                end_point = np.array([xx,yy])
                if np.linalg.norm(end_point-self.origin)<np.linalg.norm([self.ix,self.iy]-self.origin):
                    cv2.rectangle(frame,(x,y),(self.ix,self.iy),(0,255,0),3)
                elif np.linalg.norm(end_point-self.origin)>np.linalg.norm([self.ix,self.iy]-self.origin):
                    cv2.rectangle(frame,(self.ix,self.iy),(x,y),(0,255,0),3)

        while(True):
            cv2.namedWindow('Choose square meter base')
            cv2.setMouseCallback('Choose square meter base',draw_square)
            cv2.imshow('Choose square meter base',frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            elif k == ord('a'): # Add heads
                text = "Reference OK!"
                cv2.rectangle(frame,(self.ix,self.iy),(xx,yy),(0,0,255),3)
                cv2.putText(frame, text, (self.ix , self.iy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                x_coords = np.array([self.ix,xx,xx,self.ix])
                y_coords = np.array([self.iy,self.iy,yy,yy])
                square_area = 0.5*np.abs(np.dot(x_coords,np.roll(y_coords,1))-np.dot(y_coords,np.roll(x_coords,1)))   # shoelace formula
                print(square_area)

        cv2.destroyAllWindows()
        return(square_area)
