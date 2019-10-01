from imutils import face_utils
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
    ap.add_argument("-p", "--shape-predictor", required=True,
                    help="path to facial landmark predictor")
    ap.add_argument("-i", "--image", required=True,
                    help="path to input image")
    ap.add_argument("-f", "--feature", required=True,
                    help="wanted face feature")
    return vars(ap.parse_args())


def get_face_features(predictor, image, wanted_part):
    '''
    Given input image and a pre-trained shape-detector, 
    returns coordinates of facial landmarks of each detected face as coordinates with 
    values from 0 to 1000
    predictor: a trained face detector (recommended to use dlibs 68 landmark detector)
    image: input image
    wanted_part: a string dictating which facial feature that is wanted
    '''
    # TODO: make this work with multiple faces. Currently goes out of bounds after 3 faces
    image_name = image
    directory = r'.'
    os.chdir(directory)
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor)

    # load the input image, resize it, and convert to grayscale
    image = cv2.imread(image)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    faces = 0
    for(i, rect) in enumerate(rects):
        faces += 1

    get_all_features = False
    # list of wanted face values
    faces_landmarks_collector = list(np.array(np.array((int, int))))

    # range for face features are as follows
    #  0 - 16: Shape of head from left temple to right temple
    # 17 - 21: Left Eye Brow
    # 22 - 26: Right Eye Brow
    # 27 - 35: Shape of Nose
    # 36 - 41: Left Eye
    # 42 - 47: Right eye
    # 48 - 59: Outer Lips
    # 60 - 67: Inner Lips
    wanted_part = str.lower(wanted_part)
    if(wanted_part == "head"):
        start_index, end_index = 0, 16
    elif(wanted_part == "left eye brow"):
        start_index, end_index = 17, 21
    elif(wanted_part == "right eye brow"):
        start_index, end_index = 22, 26
    elif(wanted_part == "nose"):
        start_index, end_index = 27, 35
    elif(wanted_part == "left eye"):
        start_index, end_index = 36, 41
    elif(wanted_part == "right eye"):
        start_index, end_index = 42, 47
    elif(wanted_part == "eyes"):
        start_index, end_index = 36, 47
    elif(wanted_part == "lips"):
        start_index, end_index = 48, 67
    elif(wanted_part == "all"):
        wanted_part = "head"
        start_index, end_index = 0, 16
        get_all_features = True
    else:
        print("""Input string must be one of the following
        "head": Shape of head from left temple to right temple
        "left eye brow": Left Eye Brow
        "right eye brow": Right Eye Brow
        "nose": Shape of Nose
        "left eye": Left Eye
        "right eye": Right eye
        "lips": Lips
        "all" : get all features""")
        start_index, end_index = 0, 67

    # loop over detected faces
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # get center of face so that we can rotate
        (img_w, img_h) = image.shape[:2]
        center = (img_w/2, img_h/2)

        # Shape 0 and 16 values are the sides of the head right under forehead
        (ang_x, ang_y) = shape[0] - shape[16]
        angle = math.atan(ang_y/ang_x) * 180/math.pi

        # rotate the face
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image_rotate = cv2.warpAffine(image, M, (img_h, img_w))
        image_rotate = imutils.resize(image_rotate, width=500)

        # Do calculations for faces again on the rotated face
        gray_rotate = cv2.cvtColor(image_rotate, cv2.COLOR_BGR2GRAY)
        rects_rotate = detector(gray_rotate)
        # adjust shape values to the rotated image
        while(True):
            shape = predictor(gray_rotate, rects_rotate[i])
            shape = face_utils.shape_to_np(shape)
            # values that will be used to find the perfect values for cropping face
            face_x, face_y = sys.maxsize, sys.maxsize
            face_w, face_h = 0, 0
            # loop over the (x,y)-coordinates for the facial landmarks
            # and draw them on the image
            for(x, y) in shape[start_index:end_index]:
                # cv2.circle(image_rotate, (x, y), 1, (0, 0, 255), -1)
                if(x < face_x):
                    face_x = x
                if(y < face_y):
                    face_y = y
            for(x, y) in shape[start_index:end_index]:
                if(face_w < x - face_x):
                    face_w = x - face_x
                if(face_h < y - face_y):
                    face_h = y - face_y
            # crop out faces
            cropped_face = image_rotate[face_y: face_y +
                                        face_h, face_x:face_x + face_w]
            # cv2.imshow("cropped {}".format(wanted_part), cropped_face)
            # cv2.waitKey(0)
            filename = "crops/{}".format(
                wanted_part) + str(i) + "_{}".format(image_name[6:])
            cv2.imwrite(filename, cropped_face)

            # normalize images to be values from 0 to 1000
            for j in range(0, int((shape.size)/2)):
                (x, y) = shape[j]
                x_norm = (x - face_x)/(face_w) * 1000
                y_norm = (y - face_y)/(face_h) * 1000
                shape[j] = (x_norm, y_norm)

            # display an image of the wanted normalized values

            # norm_face_spots = cv2.imread("black.png")
            for j in range(start_index, end_index):
                (x, y) = shape[j]
                # cv2.circle(norm_face_spots, (x, y), 1, (255, 255, 255), -1)
            #cv2.imshow("normalized", norm_face_spots)

            # Add the coordinates of all the wanted landmarks to a list
            faces_landmarks_collector.append(shape[start_index:end_index])
            # cv2.waitKey(0)

            # if all features are wanted then cycle through the groups
            if(not get_all_features):
                break
            if(wanted_part == "head"):
                start_index, end_index = 17, 21
                wanted_part = "left eye brow"
                next
            elif(wanted_part == "left eye brow"):
                start_index, end_index = 22, 26
                wanted_part = "right eye brow"
                next
            elif(wanted_part == "right eye brow"):
                start_index, end_index = 27, 35
                wanted_part = "nose"
                next
            elif(wanted_part == "nose"):
                start_index, end_index = 36, 41
                wanted_part = "left eye"
                next
            elif(wanted_part == "left eye"):
                start_index, end_index = 42, 47
                wanted_part = "right eye"
                next
            elif(wanted_part == "right eye"):
                start_index, end_index = 48, 67
                wanted_part = "lips"
                next
            # lips will be the last one, setting the wanted_part value to head just in case more faces are wanted
            elif(wanted_part == "lips"):
                wanted_part = "head"
                break

    return faces_landmarks_collector


if __name__ == '__main__':
    args = build_arg_parser()
    shape_predictor = args["shape_predictor"]
    image = args["image"]
    feature = args["feature"]
    print(get_face_features(shape_predictor, image, feature))
