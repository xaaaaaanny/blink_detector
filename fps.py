#!/usr/bin/env python
 
import cv2
import time

cap = cv2.VideoCapture(1) #Capture video from storage/ laptop camera / IP based camera

while True:
    ret, frame = cap.read()  # read frame/image one by one     
    
    fps = int(cap.get(cv2.CAP_PROP_FPS)) # access FPS property
    print("fps:", fps) # print fps
 
    
    font = cv2.FONT_HERSHEY_SIMPLEX  #font to apply on text
    cv2.putText(frame, str(fps), (50, 50), font, 1, (0, 0, 255), 2) # add text on frame
    cv2.imshow("Live Streaming", frame)   # display frame/image
    
    key = cv2.waitKey(1)  # wait till key press 
    if key == ord("q"):  # exit loop on 'q' key press
        break
        
cap.release() # release video capture object
cv2.destroyAllWindows()  # destroy all frame windows