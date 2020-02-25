""" Experiment with face detection and image filtering using OpenCV """

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# instantiate the face detector, load XML file that describes
# the faces the detector is looking for
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# create kernel (numpy matrix) to control blurring
# (the larger the matrix, the more blurring)
# this is a rectangular structuring element
kernel = np.ones((40, 40), 'uint8')


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # run the face detector to get a list of faces in the image
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20, 20))
    for (x, y, w, h) in faces:

        # use kernal to blur the image
        frame[y:y+h, x:x+w, :] = cv2.dilate(frame[y:y+h, x:x+w, :], kernel)

        # # draw a red box around each detected face
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))

        # draw a cartoon face

        # right eye
        cv2.circle(frame[y:y+h, x:x+w, :], (int(w/12) + int(w/4), int(h/2.75)), 30, (255, 255, 255), -1)
        cv2.circle(frame[y:y+h, x:x+w, :], (int(w/4), int(h/2.75)), 15, (0, 0, 0), -1)

        # left eye
        cv2.circle(frame[y:y+h, x:x+w, :], (int(w/12) + int(w/2) + int(w/4), int(h/2.75)), 30, (255, 255, 255), -1)
        cv2.circle(frame[y:y+h, x:x+w, :], (int(w/2) + int(w/4), int(h/2.75)), 15, (0, 0, 0), -1)

        # smile
        # cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color[, thickness[, lineType[, shift]]])
        cv2.ellipse(frame[y:y+h, x:x+w, :], (int(w/2), int(h/4) + int(h/2)), (int(w/6), int(h/8)),
                   0, 0, 180, (0, 0, 255), 5)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
