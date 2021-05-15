import cv2
import numpy as np
import dlib
import random

from vid_cam import FrontCam

# https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized



detector = dlib.get_frontal_face_detector()
# detector = dlib.cnn_face_detection_model_v1("trained_models/mmod_human_face_detector.dat")
# predictor

vid_capturer = cv2.VideoCapture('screen_cap.mov')

scale_factor = 0.5

x, y, x1, y1 = 0, 0, 0, 0

num_frame = 0

while True:

    _, frame = vid_capturer.read()
    num_frame += 1

    if frame is not None:
        # don't detect on all frames
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = image_resize(gray, height=128)
        
        # only detect for every 3rd frame
        
        faces = detector(gray, 0)

        for face in faces:
            x, y = face.left(), face.top()
            x1, y1 = face.right(), face.bottom()

            cv2.rectangle(gray, (x, y), (x1, y1), (0, 255, 0), 2)

        # scaled_size = int(gray.shape[0]/scale_factor), int(gray.shape[1]/scale_factor)
        # gray = image_resize(gray, height=1000)

        cv2.imshow("Frame", gray)

    key = cv2.waitKey(1)
    if key == 27: #esc
        break