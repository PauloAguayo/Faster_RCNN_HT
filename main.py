import numpy as np
import os.path
import time
import argparse
import cv2
from sort import Sort
from detector import GroundTruthDetections

def main():
    args = vars(parse_args())
    print("[INFO] loading model...")
    detector = GroundTruthDetections(args["labels"], args["model"], args["threshold"])
    cap = cv2.VideoCapture(args["input"])
    resized = (int(args["resize"].split(',')[0]),int(args["resize"].split(',')[1]))

    if args["record"] != None:
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter(args["record"],fourcc, 20.0, (resized[0],resized[1]))

    tracker = Sort() #create instance of the SORT tracker
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (resized[0],resized[1]), interpolation = cv2.INTER_AREA)
        photo_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_expanded = np.expand_dims(photo_rgb, axis=0)

        width = frame.shape[1]
        height = frame.shape[0]
        detections = detector.get_detected_items(image_expanded,width,height)

        dets = []
        for box in detections:
            w = box[3]-box[1]
            h = box[2]-box[0]
            dets.append([box[0],box[1],box[2],box[3]]) #ymin, xmin, ymax, xmax
        trackers = tracker.update(dets,frame) #dets
        for d in trackers:
            cv2.rectangle(frame, (int(d[1]), int(d[0])), (int(d[3]), int(d[2])), (0, 255,0),0)
            text = "ID {}".format(d[4])
            cv2.putText(frame, text, (int(d[1]) , int(d[0])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

        if args["record"] != None:
            out.write(frame)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if args["record"] != None:
        out.release()
    cv2.destroyAllWindows()

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model", required=True, help="path to TensorFlow Lite object detection model")
    parser.add_argument("-l", "--labels", required=True, help="path to labels file")
    parser.add_argument("-i", "--input", default=0, type=str, help="path to optional input video file")
    parser.add_argument("-c", "--threshold", type=float, default=0.4, help="minimum probability to filter weak detection")
    parser.add_argument("-r", "--resize", type=str, default="1080,640", help="Resizing the input frame")
    parser.add_argument("-rec", "--record", type=str, default=None, help="option for recording results")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
