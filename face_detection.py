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
    ap.add_argument("-p", "--shape-predictor", required=False,
                    help="path to facial landmark predictor", default="shape_predictor_194_face_landmarks.dat")
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
    # image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    faces = 0
    for(i, rect) in enumerate(rects):
        faces += 1
    # list of wanted face values
    faces_landmarks_collector = list(np.array(np.array((int, int))))

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
    head_list = [x for x in range(11)] + \
        annoying_points + [x for x in range(114, 135)]
    lip_list = [x for x in range(
        11, 25) if x not in annoying_points] + [x for x in range(152, 194)]
    r_eye_list = [x for x in range(26, 48) if x not in annoying_points]
    l_eye_list = [x for x in range(48, 70) if x not in annoying_points]
    r_eyebrow_list = [x for x in range(70, 92) if x not in annoying_points]
    l_eyebrow_list = [x for x in range(92, 114) if x not in annoying_points]
    nose_list = range(135, 152)
    partnames_list = None
    if(wanted_part == "head"):
        wanted_list.append(head_list)
    elif(wanted_part == "lips"):
        wanted_list.append(lip_list)
    elif(wanted_part == "right eye"):
        wanted_list.append(r_eye_list)
    elif(wanted_part == "left eye"):
        wanted_list.append(l_eye_list)
    elif(wanted_part == "right eye brow"):
        wanted_list.append(r_eyebrow_list)
    elif(wanted_part == "left eye brow"):
        wanted_list.append(l_eyebrow_list)
    elif(wanted_part == "nose"):
        wanted_list.append(nose_list)
    elif(wanted_part == "all"):
        wanted_list = [head_list, lip_list, r_eye_list,
                       l_eye_list, r_eyebrow_list, l_eyebrow_list, nose_list]
        partnames_list = ["head", "lips", "right eye",
                          "left eye", "right eye brow", "left eye brow", "nose"]
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
        return None

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

        # Shape 0 and 134 values are the sides of the head right under forehead
        (ang_x, ang_y) = shape[0] - shape[134]
        angle = math.atan(ang_y/ang_x) * 180/math.pi

        # rotate the face
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image_rotate = cv2.warpAffine(image, M, (img_h, img_w))
        # image_rotate = imutils.resize(image_rotate, width=500)

        # Do calculations for faces again on the rotated face
        gray_rotate = cv2.cvtColor(image_rotate, cv2.COLOR_BGR2GRAY)
        rects_rotate = detector(gray_rotate)
        # adjust shape values to the rotated image
        while(wanted_list.__len__() > 0):
            part_list = wanted_list.pop(0)
            shape = predictor(gray_rotate, rects_rotate[i])
            shape = face_utils.shape_to_np(shape)
            # values that will be used to find the perfect values for cropping face
            face_x, face_y = sys.maxsize, sys.maxsize
            face_w, face_h = 0, 0
            # loop over the (x,y)-coordinates for the facial landmarks
            # and draw them on the image
            for(x, y) in shape[part_list]:
                # cv2.circle(image_rotate, (x, y), 1, (0, 0, 255), -1)
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
            cropped_face = image_rotate[face_y-10: face_y +
                                        face_h+10, face_x-10:face_x + face_w+10]
            # cv2.imshow("cropped {}".format(wanted_part), cropped_face)
            # cv2.waitKey(0)
            if(partnames_list is not None):
                wanted_part = partnames_list.pop(0)
            out_directory = f"crops/{wanted_part}"
            if not os.path.exists(out_directory):
                os.makedirs(out_directory)
            filename = out_directory + f"/{image_name[6:]}"
            cv2.imwrite(filename, cropped_face)

            # normalize images to be values from 0 to 1000
            for j in range(0, int((shape.size)/2)):
                (x, y) = shape[j]
                x_norm = (x - face_x)/(face_w) * 1000
                y_norm = (y - face_y)/(face_h) * 1000
                shape[j] = (x_norm, y_norm)

            # display an image of the wanted normalized values

            # norm_face_spots = cv2.imread("black.png")
            for j in part_list:
                (x, y) = shape[j]
                # cv2.circle(norm_face_spots, (x, y), 1, (255, 255, 255), -1)
            # cv2.imshow("normalized", norm_face_spots)

            # Add the coordinates of all the wanted landmarks to a list
            faces_landmarks_collector.append(shape[part_list])

    return faces_landmarks_collector


if __name__ == '__main__':
    args = build_arg_parser()
    shape_predictor = args["shape_predictor"]
    image = args["image"]
    feature = args["feature"]
    print(get_face_features(shape_predictor, image, feature))
