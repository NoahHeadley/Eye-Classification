import cv2
import numpy as np
import argparse
import os
import keyboard
from face_utils import normalize_circle


directory = os.fsencode("crops")
directory2 = os.fsencode("faces")


def locate_iris(image):

    img = cv2.imread(image)

    rows, cols, _ = img.shape
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (7, 7), 0)

    _, threshold = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    cv2.imshow("contours", threshold)
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        # cv2.drawContours(img, [cnt], -1, (0, 0, 255), 3)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        break
    cv2.imshow("iris", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Locate Iris using hough transformation
def hough_locate_iris(image):
    # Read Image as Gray-Scale
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    # Convert to gray-scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur image to reduce noise
    img_blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Apply Hough Transform on the image
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1,
                               10, param1=30, param2=10, minRadius=15, maxRadius=25)

    radius = None
    center = None
    offset = None
    # Draw Detected Circles
    if circles is not None:
        detected = True
        circles = np.uint16(np.around(circles))
        for i in circles[0, : 1, ]:
            # i[0] = x coordinate of center
            # i[1] = y coordinate of center
            # i[2] = radius of circle
            center = (i[0], i[1])
            radius = i[2]
            # Draw outer circle
            cv2.circle(img, center, radius, (0, 255, 0), 1)
            # Draw inner circle
            cv2.circle(img, center, 1, (0, 0, 255), 1)
        cv2.imshow("iris", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        center, radius, offset = normalize_circle(
            img, center, radius, maxRadius=25, minRadius=17)
    if center is not None:
        return center, radius, offset


if __name__ == '__main__':
    iris = []
    num_counted = 0
    rejects = 0
    total = 0
    for file in os.listdir(directory):
        total += 1
        filename = os.fsdecode(file)
        if filename.__contains__("eye") and not filename.__contains__("brow"):
            values = hough_locate_iris("crops/" + filename)
            if keyboard.is_pressed('x'):
                break
            if values is not None and not keyboard.is_pressed('z'):
                iris.append(values)
            else:
                rejects += 1

    print(iris)
    if(rejects is not 0):
        print(f"The number of rejects is {rejects} :(")
