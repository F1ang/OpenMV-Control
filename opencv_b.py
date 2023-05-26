#OpenCV进阶学习

#1,基础回顾
import cv2 as cv
import numpy as np

def cv_show(name,img):
    cv.imshow(name,img)

cool_girl = cv.imread('cool_girl.jpg', 1)
cv.namedWindow('cool_girl',0)
cv.resizeWindow('cool_girl',480,640)
while True:
    #roi
    roi = cool_girl[100:500,100:500]
    #通道分解与合并
    b,g,r = cv.split(cool_girl)
    b[0:200] = 0
    img = cv.merge((b,g,r))
    #对通道
    img_cope = img.copy()
    img_cope[:,:,1] = 0#g通道

    cv_show('roi', img)
    cv_show('merge', roi)
    cv_show('channel', img_cope)
    cv_show('cool_girl',cool_girl)
    key = cv.waitKey(0)
    if (key & 0xff) == ord('e'):
        break
cv.destroyAllWindows()




