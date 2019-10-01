import cv2
import numpy as np
import argparse
import os
import keyboard


directory = os.fsencode("crops")


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True)
    return vars(ap.parse_args())


def locate_iris(image):

    img = cv2.imread(image)

    rows, cols, _ = img.shape
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (7, 7), 0)

    _, threshold = cv2.threshold(gray_img, 65, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    cv2.imshow("contours", threshold)
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        #cv2.drawContours(img, [cnt], -1, (0, 0, 255), 3)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        break
    cv2.imshow("iris", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    #args = build_arg_parser()
    #image = args["image"]
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.__contains__("eye") and not filename.__contains__("brow"):
            locate_iris("crops/" + filename)
            if keyboard.is_pressed('x'):
                break
