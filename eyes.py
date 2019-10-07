import cv2
import numpy as np
import argparse
import os
import keyboard


directory = os.fsencode("crops")
directory2 = os.fsencode("faces")


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


# Locate Iris using hough transformation
def hough_locate_iris(image):
    # Read Image as Gray-Scale
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    # Convert to gray-scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur image to reduce noise
    img_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply Hough Transform on the image
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1,
                               img.shape[0]/64, param1=50, param2=10, minRadius=15, maxRadius=30)

    detected = False
    # Draw Detected Circles
    if circles is not None:
        detected = True
        circles = np.uint16(np.around(circles))
        for i in circles[0, :1, ]:
            # Draw outer circle
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 1)
            # Draw inner circle
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 1)
        cv2.imshow("iris", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return detected


if __name__ == '__main__':
    #args = build_arg_parser()
    #image = args["image"]
    num_counted = 0
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.__contains__("eye") and not filename.__contains__("brow"):
            if hough_locate_iris("crops/" + filename):
                num_counted += 1
            if keyboard.is_pressed('x'):
                break
    print(num_counted)
    # for file in os.listdir(directory2):
    #     filename = os.fsdecode(file)
    #     hough_locate_iris("faces/" + filename)
    #     if keyboard.is_pressed('x'):
    #         break
