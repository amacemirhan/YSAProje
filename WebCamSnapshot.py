# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 15:07:15 2022

@author: amacemirhan
"""

import cv2
import time
# initialize the camera
i = 0
cam = cv2.VideoCapture(0)   # 0 -> index of camera

time.sleep(2)
while(i<40):
    s, img = cam.read()
    if s:    # frame captured without any errors
        cv2.imshow('Kamera', img)
        i+=1
        cv2.imwrite("Desktop/deneme/filename("+str(i)+").jpg",img) #save image
        time.sleep(0.7)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cam.release()
 # Destroy all the windows
cv2.destroyAllWindows()