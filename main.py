import Iris_Detector
import face_detection as fd
import argparse
import os


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", required=False,
                    help="path to facial landmark predictor", default="shape_predictor_194_face_landmarks.dat")
    ap.add_argument("-i", "--image", required=True,
                    help="path to input image")
    return vars(ap.parse_args())


def get_eye_info(right_eye, left_eye, image):
    #right_eye = fd.get_face_features(shape_predictor, image, 'right eye')
    area_1 = fd.find_area(right_eye)
    iris_info_1 = Iris_Detector.get_iris(f"crops/right_eye/{image[6:]}")
    print(area_1, iris_info_1)

    area_2 = fd.find_area(left_eye)
    iris_info_2 = Iris_Detector.get_iris(f"crops/left_eye/{image[6:]}")

    print(area_2, iris_info_2)


if __name__ == '__main__':
    args = build_arg_parser()
    shape_predictor = args["shape_predictor"]
    image = args["image"]
    face = fd.get_face_features(shape_predictor, image, 'all')
    head = face[0]
    lips = face[1]
    top_lip = face[2]
    bottom_lip = face[3]
    right_eye = face[4]
    left_eye = face[5]
    right_eye_brow = face[6]
    left_eye_brow = face[7]
    nose = face[8]
    get_eye_info(right_eye, left_eye, image)
