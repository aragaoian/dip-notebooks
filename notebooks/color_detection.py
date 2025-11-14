import cv2
import numpy as np
import time

cap = cv2.VideoCapture('/dev/video0')
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

time.sleep(1)
count = 0
background = 0

for i in range(60):
    check, background = cap.read()
    if not check: break
background = np.flip(background, axis=1)

if not cap.isOpened():
    print('Error')
else:
    while True:
        check, frame = cap.read()
        if not check: break

        count +=1
        frame = np.flip(frame, axis=1) # create mirror effect
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  

        # setting the lower and upper range for green color (detecting green)
        lower_green = np.array([40, 40, 40])        
        upper_green = np.array([70, 255, 255]) 
        mask1 = cv2.inRange(hsv, lower_green, upper_green)

        # setting the lower and upper range for another shade of green 
        lower_green = np.array([35, 40, 40]) 
        upper_green = np.array([85, 255, 255]) 
        mask2 = cv2.inRange(hsv, lower_green, upper_green)

        mask1 = mask1 + mask2 # highlights all detected green areas in the image.
  
        # Refining the mask corresponding to the detected red color 
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), 
                                            np.uint8)) # opening op (erosion + dilation), removes small white noises from the mask
        mask1 = cv2.dilate(mask1, np.ones((3, 3), np.uint8)) # make the white areas bigger and therefore making the detected green areas more prominent.
        mask2 = cv2.bitwise_not(mask1) # inverted mask to segment out the green color
    
        # Generating the final output 
        res1 = cv2.bitwise_and(background, background, mask = mask1) # green area will be retained and everything else will be black, 
                                                                     # meaning it will ignore green pixels and replace with the background pixel
        res2 = cv2.bitwise_and(frame, frame, mask = mask2) # only non-green parts of the original frame are shown
        final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

        cv2.imshow('video', final_output)
        key = cv2.waitKey(1)
        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()
