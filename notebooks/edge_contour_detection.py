import cv2
import numpy as np
# print("OpenCV version:", cv2.__version__)

cap = cv2.VideoCapture("/dev/video0")
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) # This line configures the codec that will be used to decode the video stream coming from the camera.
# MJPEG was choosen because of it is useful for real-time capture (and also because imshow() was having timeout problems due to this)

if not cap.isOpened():
    print("Error: Could not open video device")
else:
    while True:
        check, frame = cap.read()
        if not check: break # check returns a bool showing if the camera reading was sucessful or not

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_frame, 100, 200, 3, L2gradient=True) # Popular edge detection algorithm
        # Noise reduction; Finding intensity gradient of the image; 
        # Non-maximum Suppression (removing unwanted pixels); Hysteresis Thresholding (decides which are all edges are really edges and which are not.)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, 
                                               cv2.CHAIN_APPROX_NONE)
        # contours -> list of all contours retrived from the image
        # hierarchy ->  a hierarchy of contours that provides information about the image's topology (unsed because of RETR_EXTERNAL)
        # RETR_EXTERNAL -> retrieve only the outermost contours (external ones), not considering nested contours
        # CHAIN_APPROX_NONE -> contours approximation method. In this case all points will be stored (consumes a lot of memory)
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 3) # This does the same thing as using cv2.Canny(), finding contours from edges
                                                              # (which is essentialy the edges canny already detected)
        cv2.imshow('video', edges)

        key = cv2.waitKey(1)
        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()
