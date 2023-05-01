#####
##### Ф.И: Ноурузи Мехди
##### ИСУ: 317306
##### группа: R3135
#####Номер варианта: 6
#####
#Задача 1

import cv2

img = cv2.imread('images/variant-6.png')
resized_img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

cv2.imshow('Resized image', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()