#####
##### Ф.И: Ноурузи Мехди
##### ИСУ: 317306
##### группа: R3135
#####Номер варианта: 6
#####
#Задача 2

import cv2
import numpy as np
import time


ref_image = cv2.imread('ref-point.jpg', cv2.IMREAD_GRAYSCALE)
threshold = 0.8
cap = cv2.VideoCapture(0)
count_left = 0
count_right = 0
prev_count_left = 0
prev_count_right = 0
detection_timer_left = 0
detection_timer_right = 0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
half_width = int(width / 2)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray, ref_image, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    
    for pt in zip(*loc[::-1]):
        if pt[0] < half_width:
            cv2.rectangle(frame, pt, (pt[0]+ref_image.shape[1], pt[1]+ref_image.shape[0]), (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, pt, (pt[0]+ref_image.shape[1], pt[1]+ref_image.shape[0]), (255, 0, 0), 2)
    
    percentage = round(np.max(res) * 100, 2)
    
    if percentage >= threshold*100 and np.any(loc[1] < half_width):
        detection_timer_left += 1
        if detection_timer_left >= 30: # 1 second
            count_left += 1
            prev_count_left = count_left
            detection_timer_left = 0
    else:
        detection_timer_left = 0
    
    if percentage >= threshold*100 and np.any(loc[1] >= half_width):
        detection_timer_right += 1
        if detection_timer_right >= 30: # 1 second
            count_right += 1
            prev_count_right = count_right
            detection_timer_right = 0
    else:
        detection_timer_right = 0
    
    cv2.putText(frame, f'{percentage}% Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'Left : {count_left}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'Right : {count_right}', (half_width+50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Image Detector', frame)
    

    if cv2.waitKey(1) & 0xFF == ord('e'):  # Press 'e' and Exit 
        break