import tensorflow as tf
import numpy as np
from utils import label_map_util

class GroundTruthDetections:

    def __init__(self, label, model, threshold):             #### INFERENCIAS

        self.classes = 2
        self.label_map = label_map_util.load_labelmap(label)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

        # Load the Tensorflow model into memory.
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            self.od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model, 'rb') as fid:
                self.serialized_graph = fid.read()
                self.od_graph_def.ParseFromString(self.serialized_graph)
                tf.import_graph_def(self.od_graph_def, name='')
            self.sess = tf.Session(graph=self.detection_graph)

        # Define input and output tensors (i.e. data) for the object detection classifier
        # Input tensor is the image
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

        # Number of objects detected
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        self.threshold = threshold



    '''as in practical realtime MOT, the detector doesn't run on every single frame'''
    def _do_detection(self, detect_prob = .6):
        return int(np.random.choice(2, 1, p=[1 - detect_prob, detect_prob]))

    '''returns the detected items positions or [] if no detection'''
    def get_detected_items(self,photo_expanded,width,height):

        if self._do_detection():

            (boxes, scores, classes, num) = self.sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: photo_expanded})

            boxes = boxes.reshape(boxes.shape[1],4)
            scores = scores.reshape(scores.shape[1],1)
            classes = classes.reshape(classes.shape[1],1)

            objs_detected = []
            for box,score,clas in zip(boxes,scores,classes):
                if score >= self.threshold and clas == 1.0:
                    objs_detected.append([box[0]*height,box[1]*width,box[2]*height,box[3]*width])#,score])  #ymin,xmin,ymax,xmax,score
            return (np.array(objs_detected))
        else:
            return []
