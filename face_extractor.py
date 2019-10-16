import os
import face_detection
import argparse
import numpy as np

directory = os.fsencode("faces")
collection = list(np.array(np.array((int, int))))
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=False,
                help="path to facial landmark predictor", default="shape_predictor_194_face_landmarks.dat")
ap.add_argument("-f", "--feature", required=True,
                help="wanted face feature")
args = vars(ap.parse_args())

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    predictor = args["shape_predictor"]
    wanted_feature = args["feature"]
    collection.extend(face_detection.get_face_features(
        predictor, "faces/" + filename, wanted_feature))
print(collection)
