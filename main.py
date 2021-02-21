import numpy as np
import argparse
import cv2
from sort import Sort
from detector import GroundTruthDetections
import time
import csv
from shapely import geometry
from real_time import Real_time
from calibrate import Calibration
from draw import Drawing


def main():
    args = vars(parse_args())
    print("[INFO] LOADING MODEL...")
    detector = GroundTruthDetections(args["labels"], args["model"], args["threshold"])
    cap = cv2.VideoCapture(args["input"])

    video_length = int(cv2.VideoCapture.get(cap, int(cv2.CAP_PROP_FRAME_COUNT)))

    resized = (int(args["resize"].split(',')[0]),int(args["resize"].split(',')[1])) # x,y
    csv_name = str(time.asctime(time.localtime()).replace(" ","_").replace(":","."))
    if args["calibration"] == True:
        calibration = Calibration([resized[0],resized[1]])
        calibration.Checkboard()
    real_plot = Real_time()

    if args["record"] != None:
        from record import Record_video
        if args["chart"]:
            csv_name = "charted_"+args["record"].split('.')[0]+"_"+csv_name
            save_video = Record_video("Results/charted_"+args["record"], resized, args["chart"])
        else:
            csv_name = args["record"].split('.')[0]+"_"+csv_name
            save_video = Record_video("Results/"+args["record"], resized, args["chart"])

    people_height = args["people-height"]
    camera_height = args["camera-height"]

    start_time = time.time()
    fields = ['Time','Frame','New_detections','Current_detections','Total_detections','Density (people/m2)']
    draws = Drawing(resized) ##################################################3
    with open("Results/"+csv_name+".csv",'w',newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(fields)

        tracker = Sort()
        real_counting = 0
        total_detections = 0
        total_ids = []
        frames = 0
        var = True
        while frames<video_length:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (resized), interpolation = cv2.INTER_AREA)
            if args["calibration"] == True:
                frame = calibration.Undistort(frame)

            if var:
                # Get polygon and centroid points
                pts = np.array(draws.Generate_Polygon('Polygon Image',frame), np.int32)
                asking = input('Do you know polygon area? (y/n)')
                B = True
                if asking == 'y':
                    area_polygon = input('Enter the area (m2) ')
                    B = False
                while(B):
                    square_slf = draws.Square_meter(frame)
                    area_slf = 0.5*np.abs(np.dot(pts[:,0],np.roll(pts[:,1],1))-np.dot(pts[:,1],np.roll(pts[:,0],1)))
                    area_polygon = area_slf/square_slf
                    print('AREA POLIGONO =', float(area_polygon),'m2')
                    asking = input('Is the area measure correct? (y/n)')
                    if asking == 'y':
                        break
                poly = geometry.Polygon(pts) # x,y format
                polygon_centroid = np.array(list(poly.centroid.coords)[0]) # x,y
                var = False

            photo_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_expanded = np.expand_dims(photo_rgb, axis=0)

            detections = detector.get_detected_items(image_expanded,resized[0],resized[1])

            dets = []
            for box in detections:
                dets.append([box[0],box[1],box[2],box[3]]) #ymin, xmin, ymax, xmax
            trackers = tracker.update(dets,frame) #dets
            new_detections = 0
            current_detections = 0
            for d in trackers:
                centroid_box = np.array([int(d[1]+(d[3]-d[1])/2),int(d[0]+(d[2]-d[0])/2)]) # x,y
                length_box = np.array([d[3]-d[1],d[2]-d[0]])
                ideal_point = draws.Draw_detections(frame,centroid_box,length_box,camera_height,people_height)
                point = geometry.Point(ideal_point[1],ideal_point[0])
                if poly.contains(point):
                    cv2.rectangle(frame, (int(d[1]), int(d[0])), (int(d[3]), int(d[2])), (0, 255,0),0)
                    text = "ID "+str(int(d[4]))
                    cv2.putText(frame, text, (int(d[1]) , int(d[0])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                    #cv2.circle(frame,(ideal_point[1],ideal_point[0]),4,(255,255,0),-1)
                    if int(d[4]) not in total_ids:
                        total_ids.append(int(d[4]))
                        new_detections+=1
                    current_detections+=1
            total_detections = len(total_ids)

            if args["show"]:
                text = "Detections = " + str(int(current_detections))
                cv2.putText(frame, text, (int(resized[0]/2)+50,resized[1]-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

            clock = time.time() - start_time
            frames+=1
            density = float(current_detections)/float(area_polygon)
            row = ["%s" % (clock),str(frames),str(new_detections),str(current_detections),str(total_detections),str(density)]
            csv_writer.writerow(row)

            cv2.putText(frame, 'Density = '+str(round(density,3))+' passengers/(m2)', (int(resized[0]*0.1) , resized[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if args["limit"]<current_detections:
                cv2.putText(frame, 'Platform overcrowded!', (600,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2) ################ arreglar las coordenadas

            real_plot.Online_plot(clock, current_detections, new_detections, args["limit"],args["window"])

            if args["record"] != None:
                save_video.Recording(frame)

            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        if args["record"] != None:
            save_video.End_recording()
        cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model", required=True, help="Path to TensorFlow Lite object detection model")
    parser.add_argument("-l", "--labels", required=True, help="Path to labels file")
    parser.add_argument("-i", "--input", default=0, type=str, help="Path to input video file")
    parser.add_argument("-c", "--threshold", type=float, default=0.75, help="Minimum probability to filter weak detection")
    parser.add_argument("-r", "--resize", type=str, default="1080,640", help="Resizing the input frame,e.g. 1080,640")
    parser.add_argument("-s", "--show", action="store_true", help="Show people counter")
    parser.add_argument("-g", "--chart", action="store_true", help="Show online chart in video")
    parser.add_argument("-w", "--window", type=int, default=300, help="Time window")
    parser.add_argument("-H", "--camera-height", type=float, default=2.5, help="z-coordinate for camera positioning")
    parser.add_argument("-p", "--people-height", type=float, default=1.7, help="z-coordinate for people height")
    parser.add_argument("-rec", "--record", type=str, default=None, help="Option for recording results")
    parser.add_argument("-cal", "--calibration", action="store_true", help="Option for un-distort input image")
    parser.add_argument("-lim", "--limit", type=float, default=5, help="Limit warning")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
