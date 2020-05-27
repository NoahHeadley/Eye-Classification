from imutils import face_utils
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os
import imutils
import dlib
import cv2
import math


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", required=False,
                    help="path to facial landmark predictor", default="shape_predictor_194_face_landmarks.dat")
    ap.add_argument("-i", "--image", required=True,
                    help="path to input image")
    ap.add_argument("-f", "--feature", required=True,
                    help="wanted face feature")
    return vars(ap.parse_args())


def find_area(points):
    """
    Calculates the area of the shape formed by connecting all the points

    Parameter:
    points (List(Int,Int)): A set of x,y coordinates for every landmark 
    """

    try:
        # This will only work in a list of coordinates
        # get all the x and y coordinates
        x = points[:, 0]
        y = points[:, 1]
        x_super = np.array([])
        y_super = np.array([])
        for i in range(y.size):
            x1 = np.linspace(x[i], x[(i+1) % x.size],
                             num=100)
            y1 = np.linspace(y[i], y[(i+1) % y.size],
                             num=100)
            x_super = np.append(x_super, x1)
            y_super = np.append(y_super, y1)
            # print(x1)
            # print(y1)

            #plt.plot(x1, y1, '-o')
            # plt.show()
        x_super.flatten()
        y_super.flatten()
        size = np.size(x_super)
        area = 0
        for i in range(int(size/2)):
            height = np.abs(y_super[i] - y_super[size - i - 1])
            # width = np.abs((x_super[i] - x_super[size - i - 1]))
            area += height
        return area
    except Exception as e:
        print(e)


def find_angle(points):
    return None


def get_face_features(predictor, image, wanted_part):
    '''
    Given input image and a pre-trained shape-detector,
    returns coordinates of facial landmarks of each detected face as coordinates with
    values from 0 to 1000
    predictor: a trained face detector (recommended to use dlibs 194 landmark detector)
    image: input image
    wanted_part: a string dictating which facial feature that is wanted
    '''
    image_name = image
    directory = r'.'
    os.chdir(directory)
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor)

    # load the input image, resize it, and convert to grayscale
    image = cv2.imread(image)
    # image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    faces = 0
    for(i, rect) in enumerate(rects):
        faces += 1
    # list of wanted face values
    faces_landmarks_collector = list()

    # range for face features are as follows
    #  0 - 10: Left Cheek
    # 21, 32, 43, 54, 65, 76, 87, 98, 109 - annoyingly these are the rest of the left of the head
    # 11 - 25: center of lips
    # 26 - 47: Right Eye
    # 48 - 69: Left Eye
    # 70 - 91: Right Eyebrow
    # 92 - 113: Left Eyebrow
    # 114 - 134: Right side of head
    # 135 - 152: Nose
    # 153 - 193: Rest of lips

    # So this means that head shape is [0:10] + [21,32, 43, 54, 65, 76, 87, 98, 109] + [114:134]
    # also this means that all ranges with those annoying points need to exclude them
    wanted_part = str.lower(wanted_part)
    annoying_points = [21, 32, 43, 54, 65, 76, 87, 98, 109]
    wanted_list = []
    r_eye_list = [x for x in range(26, 48) if x not in annoying_points]
    l_eye_list = [x for x in range(48, 70) if x not in annoying_points]
    partnames_list = None
    all_mode = True
    all_list = r_eye_list + l_eye_list
    if(wanted_part == "eyes"):
        wanted_list.append(r_eye_list)
        wanted_list.append(l_eye_list)
        partnames_list = ["right_eye", "left_eye"]
    else:
        print("""Input string must be one of the following
        "left eye": Left Eye
        "right eye": Right eye
        "eyes" : get both eyes""")
        return None

    # loop over detected faces
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        face_x, face_y = sys.maxsize, sys.maxsize
        face_w, face_h = 0, 0

        while(wanted_list.__len__() > 0):
            part_list = wanted_list.pop(0)
            # values that will be used to find the perfect values for cropping face
            face_x, face_y = sys.maxsize, sys.maxsize
            face_w, face_h = 0, 0
            # loop over the (x,y)-coordinates for the facial landmarks and get the coordinates containing the points
            for(x, y) in shape[part_list]:
                if(x < face_x):
                    face_x = x
                if(y < face_y):
                    face_y = y
            for(x, y) in shape[part_list]:
                if(face_w < x - face_x):
                    face_w = x - face_x
                if(face_h < y - face_y):
                    face_h = y - face_y
            # crop out faces
            cropped_face = image[face_y-30: face_y +
                                 face_h+30, face_x-30: face_x + face_w+30]
            # cv2.waitKey(0)
            if(partnames_list is not None):
                wanted_part = partnames_list.pop(0)
            out_directory = f"crops/{wanted_part}"
            if not os.path.exists(out_directory):
                os.makedirs(out_directory)
            filename = out_directory + f"/{image_name[6:]}"
            cv2.imwrite(filename, cropped_face)

            # Add the coordinates of all the wanted landmarks to a list
            shape[part_list].sort()
            faces_landmarks_collector.append(shape[part_list])

    for(x, y) in shape[all_list]:
        if(x < face_x):
            face_x = x
        if(y < face_y):
            face_y = y
    for(x, y) in shape[all_list]:
        if(face_w < x - face_x):
            face_w = x - face_x
        if(face_h < y - face_y):
            face_h = y - face_y

    final_faces = []
    for l in faces_landmarks_collector:
        for j in range(0, int((l.size)/2)):
            (x, y) = l[j]
            x_norm = (x - face_x)/(face_w) * 1000
            y_norm = (y - face_y)/(face_h) * 1000
            l[j] = (x_norm, y_norm)
        final_faces.append(l)
    return final_faces


if __name__ == '__main__':
    args = build_arg_parser()
    shape_predictor = args["shape_predictor"]
    image = args["image"]
    feature = args["feature"]
    output = list(get_face_features(shape_predictor, image, feature))
    for i in output[0]:
        print(i)
