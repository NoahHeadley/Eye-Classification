import Iris_Detector
import face_detection as fd
import argparse
import os
import math


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", required=False,
                    help="path to facial landmark predictor", default="shape_predictor_194_face_landmarks.dat")
    return vars(ap.parse_args())


def get_dist(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist


def get_eye_info(right_eye, left_eye, image):
    # right_eye = fd.get_face_features(shape_predictor, image, 'right eye')
    area_1 = fd.find_area(right_eye)
    width_1, height_1, iris_area_1, covered_1 = Iris_Detector.get_iris(
        f"crops/right_eye/{image[6:]}")

    area_2 = fd.find_area(left_eye)
    width_2, height_2, iris_area_2, covered_2 = Iris_Detector.get_iris(
        f"crops/left_eye/{image[6:]}")

    return area_1, area_2, width_1, width_2, height_1, height_2, iris_area_1, iris_area_2, covered_1, covered_2


if __name__ == '__main__':
    args = build_arg_parser()
    shape_predictor = args["shape_predictor"]

    path = 'faces/'
    directory = os.fsencode("faces")
    out_file = open('eyes.txt', 'x')
    num_variables = 11
    num_clusters = 10
    out_file.write(
        f"{os.listdir(directory).__len__()} {num_variables} {num_clusters}\n")
    for file in os.listdir(directory):
        image = path + os.fsdecode(file)
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
        area_1, area_2, width_1, width_2, height_1, height_2, iris_area_1, iris_area_2, covered_1, covered_2 = get_eye_info(
            right_eye, left_eye, image)
        dist = get_dist(right_eye[0, 0], right_eye[0, 1],
                        left_eye[0, 0], left_eye[0, 1])
        out_file.write(
            f"{area_1} {area_2} {width_1} {width_2} {height_1} {height_2} {iris_area_1} {iris_area_2} {covered_1} {covered_2} {dist}\n")
    out_file.close()
