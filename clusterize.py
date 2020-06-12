"""
Clusterize creates a file titled eyes.txt that contains all the variables
measured from the set of eyes. This program takes a long time as it has to
run the Iris detector on every image and that has to start Tensorflow everytime
"""
import Iris_Detector as iris
import face_detection as fd
import argparse
import os
import math
from datetime import datetime
from pyspark.sql import SparkSession


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", required=False,
                    help="path to facial landmark predictor", default="shape_predictor_194_face_landmarks.dat")
    return vars(ap.parse_args())


def get_dist(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist


def get_eye_info(face):
    """
    Calculates all the wanted variables for the pictures of eyes

    Parameters:
    face: a list containing the following
    right_eye (List(int,int)): Normalized coordinates of the landmarks for the right eye
    left_eye (List(int, int)): Normalized coordinates of the landmarks for the left eye
    image (String): The filename of the image
    """
    (right_eye, left_eye, image) = face
    area_1 = fd.find_area(right_eye)
    area_2 = fd.find_area(left_eye)
    width_1, height_1, iris_area_1, covered_1, width_2, height_2, iris_area_2, covered_2 = iris.get_iris(
        f"crops/right_eye/{image}", f"crops/left_eye/{image}")

    return area_1, area_2, width_1, width_2, height_1, height_2, iris_area_1, iris_area_2, covered_1, covered_2


def apache_face_decode(file):
    path = 'faces/'
    image = os.fsdecode(file)
    face = fd.get_face_features(shape_predictor, path + image, 'eyes')
    return (face, image)


if __name__ == '__main__':
    spark = SparkSession\
        .builder\
        .appName("Eye Data")\
        .getOrCreate()
    args = build_arg_parser()
    shape_predictor = args["shape_predictor"]

    path = 'faces/'
    directory = os.listdir(os.fsencode("faces"))
    out_file = open('eyes.txt', 'w')
    num_variables = 11
    num_clusters = math.ceil(math.sqrt(len(directory)))
    out_file.write(
        f"{len(directory)} {num_variables} {num_clusters}\n")
    count = 1
    total = len(directory)
    start = datetime.now()

    eye_variables = spark.sparkContext.parallelize(
        directory).map(apache_face_decode)
    eye_values = eye_variables.collect()

    faces = []
    filenames = []

    for v in eye_values:
        (face, filename) = v
        faces.append(face)
        filenames.append(filename)
    # for file in directory:
    #     image = os.fsdecode(file)
    #     face = fd.get_face_features(shape_predictor, path + image, 'eyes')
    #     faces.append(face)
    #     filenames.append(image)
    #     time_elapsed = datetime.now() - start
    #     average_time = time_elapsed/count
    #     expected = average_time * (total - count)
    #     print(
    #         f"Done: {count} / {total} = {round(float(count/total) * 100, 2)}%\tTime Elapsed:{time_elapsed}\tRemaining time:{expected}")
    #     count += 1

    eyes = []
    for x in range(total):
        eyes.append((faces[x][0], faces[x][1], filenames[x]))

    data = spark.sparkContext.parallelize(
        eyes).map(get_eye_info)

    values = data.collect()

    for v in values:
        area_1 = v[0]
        area_2 = v[1]
        width_1 = v[2]
        width_2 = v[3]
        height_1 = v[4]
        height_2 = v[5]
        iris_area_1 = v[6]
        iris_area_2 = v[7]
        covered_1 = v[8]
        covered_2 = v[9]
        out_file.write(
            f"{area_1} {area_2} {width_1} {width_2} {height_1} {height_2} {iris_area_1} {iris_area_2} {covered_1} {covered_2}\n")

    # for i in range(len(directory)):
    #     image = filenames[i]
    #     face = faces[i]
    #     right_eye = face[0]
    #     left_eye = face[1]
    #     area_1, area_2, width_1, width_2, height_1, height_2, iris_area_1, iris_area_2, covered_1, covered_2 = get_eye_info(
    #         right_eye, left_eye, image)
    #     dist = get_dist(right_eye[0, 0], right_eye[0, 1],
    #                     left_eye[0, 0], left_eye[0, 1])
    #     out_file.write(
    #         f"{area_1} {area_2} {width_1} {width_2} {height_1} {height_2} {iris_area_1} {iris_area_2} {covered_1} {covered_2} {dist}\n")
    #     time_elapsed = datetime.now() - start
    #     average_time = time_elapsed/count
    #     print(
    #         f"Done: {count} / {total} = {round(float(count/total) * 100, 2)}%\tTime Elapsed:{time_elapsed}\tAverage Time: {average_time}")
    #     count += 1

    out_file.close()
    spark.stop()
