# Some of this is copied from EdjeElectronic's Tensorflow guide (https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10)

# I have modified this to work with my system and return data

import math
import sys
import tensorflow as tf
import numpy as np
import cv2
from utils import visualization_utils as vis_util
from utils import label_map_util
import os
import argparse
import warnings


sys.path.append("..")

# This is the directory containing the eyes used for testing
directory = os.fsencode("demo")
# Import utilites

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=False,
                    help="image of an eye", default="me.jpg")
    return vars(ap.parse_args())


def get_iris(right_eye, left_eye):
    warnings.simplefilter(action='ignore', category=FutureWarning)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    IMAGE_NAME = right_eye
    IMAGE_NAME2 = left_eye

    # Grab path to current working directory
    CWD_PATH = os.getcwd()

    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(
        CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')

    # Path to image
    PATH_TO_IMAGE = os.path.join(CWD_PATH, IMAGE_NAME)
    PATH_TO_IMAGE2 = os.path.join(CWD_PATH, IMAGE_NAME2)

    # Number of classes the object detector can identify
    NUM_CLASSES = 1

    # Load the label map.
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where the iris was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    image = cv2.imread(PATH_TO_IMAGE)
    image_expanded = np.expand_dims(image, axis=0)

    image2 = cv2.imread(PATH_TO_IMAGE2)
    image2_expanded = np.expand_dims(image2, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    max_boxes_to_draw = boxes.shape[0]

    # if the confidence score is under 80% then the iris dimensions aren't trustworthy
    min_score_thresh = 0.8

    # This goes through each of the detected boxes and their associated scores and calculates relevent data for the project
    for box, score in zip(boxes, scores):
        for i in range(min(max_boxes_to_draw, boxes.shape[0])):
            if score[i] > min_score_thresh:
                (xmin, ymin, xmax, ymax) = box[i]
                radius = (xmax - xmin)/2
                area = math.pi * (radius ** 2)
                width = xmax - xmin
                height = ymax - ymin
                # This is the length of the iris that is obscured by the eyelids without regard to which end it is obscured
                covered_length = 2*radius - (ymax-ymin)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image2_expanded}
    )

    # These are the values of the second eye's iris. If the second eye's iris isn't properly detected, the values will default to the firsts
    if width != None:
        width2 = width
        height2 = height
        area2 = area
        covered_length2 = covered_length
    max_boxes_to_draw = boxes.shape[0]
    for box, score in zip(boxes, scores):
        for i in range(min(max_boxes_to_draw, boxes.shape[0])):
            if score[i] > min_score_thresh:
                (xmin2, ymin2, xmax2, ymax2) = box[i]
                radius2 = (xmax2 - xmin2)/2
                area2 = math.pi * (radius2 ** 2)
                width2 = xmax2 - xmin2
                height2 = ymax2 - ymin2
                # This is the length of the iris that is obscured by the eyelids without regard to which end it is obscured
                covered_length2 = 2*radius2 - (ymax2-ymin2)

    if width2 == None:
        print(f"Error: Iris is not found in this image: {right_eye}")
    elif width2 != None and width == None:
        width = width2
        height = height2
        area = area2
        covered_length = covered_length2

    return width, height, area, covered_length, width2, height2, area2, covered_length2


if __name__ == '__main__':
    args = build_arg_parser()
    get_iris(args['image'], args['image'])
