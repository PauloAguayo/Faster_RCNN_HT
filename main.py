import numpy as np
import os.path
import time
import argparse
import cv2
from sort import Sort
from detector import GroundTruthDetections
import time
import csv
from real_time import Real_time


def main():
    args = vars(parse_args())
    print("[INFO] LOADING MODEL...")
    detector = GroundTruthDetections(args["labels"], args["model"], args["threshold"])
    cap = cv2.VideoCapture(args["input"])

    video_length = int(cv2.VideoCapture.get(cap, int(cv2.CAP_PROP_FRAME_COUNT)))

    resized = (int(args["resize"].split(',')[0]),int(args["resize"].split(',')[1]))
    csv_name = str(time.asctime(time.localtime()).replace(" ","_").replace(":","."))

    real_plot = Real_time()

    if args["record"] != None:
        from record import Record_video
        if args["chart"]:
            csv_name = "charted_"+args["record"].split('.')[0]+"_"+csv_name
            save_video = Record_video("Results/charted_"+args["record"], resized, args["chart"])
        else:
            csv_name = args["record"].split('.')[0]+"_"+csv_name
            save_video = Record_video("Results/"+args["record"], resized, args["chart"])

    start_time = time.time()
    fields = ['Time','Frame','New_detections','Current_detections','Total_detections']
    with open("Results/"+csv_name+".csv",'w',newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(fields)

        tracker = Sort()
        real_counting = 0
        frames = 0
        while frames<video_length:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (resized), interpolation = cv2.INTER_AREA)
            photo_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_expanded = np.expand_dims(photo_rgb, axis=0)

            detections = detector.get_detected_items(image_expanded,resized[0],resized[1])

            dets = []
            for box in detections:
                dets.append([box[0],box[1],box[2],box[3]]) #ymin, xmin, ymax, xmax
            trackers = tracker.update(dets,frame) #dets
            for d in trackers:
                cv2.rectangle(frame, (int(d[1]), int(d[0])), (int(d[3]), int(d[2])), (0, 255,0),0)
                text = "ID "+str(int(d[4]))
                cv2.putText(frame, text, (int(d[1]) , int(d[0])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

            if args["show"]:
                try:
                    max_counting = np.amax(trackers[:,4])
                    current_counting = len(trackers[:,4])
                except:
                    max_counting = 0
                    current_counting = 0
                dif_counting = 0
                if real_counting<max_counting:
                    dif_counting = int(max_counting - real_counting)
                    real_counting = max_counting
                text = "Detections = " + str(int(real_counting))
                cv2.putText(frame, text, (10,resized[1]-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

            clock = time.time() - start_time
            frames+=1
            row = ["%s" % (clock),str(frames),str(dif_counting),str(current_counting),str(real_counting)]
            csv_writer.writerow(row)

            real_plot.Online_plot(clock, current_counting, dif_counting, args["window"])

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
    parser.add_argument("-g", "--chart", action="store_true", help="Show online chart")
    parser.add_argument("-w", "--window", type=int, default=300, help="Time window")
    parser.add_argument("-rec", "--record", type=str, default=None, help="Option for recording results")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
