# Some of this is copied from EdjeElectronic's Tensorflow guide (https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10)

# I have modified this to work with my system and return data

from utils import visualization_utils as vis_util
from utils import label_map_util
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import math

sys.path.append("..")

# This is the directory containing the eyes used for testing
directory = os.fsencode("demo")
# Import utilites

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
for file in os.listdir(directory):
    IMAGE_NAME = "demo/" + os.fsdecode(file)

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
    # Each box represents a part of the image where a particular object was detected
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

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
#     vis_util.visualize_boxes_and_labels_on_image_array(
#         image,
#         np.squeeze(boxes),
#         np.squeeze(classes).astype(np.int32),
#         np.squeeze(scores),
#         category_index,
#         use_normalized_coordinates=True,
#         line_thickness=1,
#         min_score_thresh=0.80)

    max_boxes_to_draw = boxes.shape[0]

    min_score_thresh = 0.80

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
                print(
                    f"width = {width}\nheight={height}\narea={area}\ncovered={covered_length}\n")
    # cv2.imshow('Object detector', image)

    # Press any key to close the image
    # cv2.waitKey(0)

    # Clean up
    # cv2.destroyAllWindows()