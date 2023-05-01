#####
##### Ф.И: Ноурузи Мехди
##### ИСУ: 317306
##### группа: R3135
#####Номер варианта: 6
#####
#Дополнительная задача

import cv2
import numpy as np

ref_image = cv2.imread('ref-point.jpg', cv2.IMREAD_GRAYSCALE)
fly_image = cv2.imread('fly64.png')
fh, fw = fly_image.shape[:2]
cap = cv2.VideoCapture(0)
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(ref_image, None)
    kp2, des2 = orb.detectAndCompute(gray, None)

    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) > 0:
        top_match = matches[0]
        ref_center_x = int(ref_image.shape[1] / 2)
        ref_center_y = int(ref_image.shape[0] / 2)
        match_center_x = int(kp2[top_match.trainIdx].pt[0])
        match_center_y = int(kp2[top_match.trainIdx].pt[1])

        distance = np.sqrt((ref_center_x - match_center_x) ** 2 + (ref_center_y - match_center_y) ** 2)
        x = match_center_x - int(fw / 2)
        y = match_center_y - int(fh / 2)
        alpha = 0.5
        beta = 1 - alpha
        roi = frame[y:y+fh, x:x+fw]
        overlay = cv2.addWeighted(roi, alpha, fly_image, beta, 0)
        frame[y:y+fh, x:x+fw] = overlay

        if np.array_equal(overlay, fly_image):
            cv2.putText(frame, "MATCH!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "NO MATCH", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    cv2.imshow('frame', frame)


    if cv2.waitKey(1) & 0xFF == ord('e'):  # Press 'e' and Exit 
        break

cap.release()
cv2.destroyAllWindows()


