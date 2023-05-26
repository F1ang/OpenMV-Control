#1,show picture and video
# import cv2 as cv
# import numpy as np
# #窗口
# cv.namedWindow('cool_girl',0)
# cv.resizeWindow('cool_girl',640,480)
# #读图片
# img = cv.imread("cool_girl.jpg")
# #video保存格式
# fourcc = cv.VideoWriter_fourcc(*'MJPG')
# viwr = cv.VideoWriter('./show.mp4',fourcc,24,(640,480))#output 编码器 帧率 分辨率
# #video读取
# cap = cv.VideoCapture(0)
#
# while cap.isOpened():
#     ret,frame = cap.read()
#     if ret == 1:
#         cv.resizeWindow('cool_girl', 640, 480)
#         cv.imshow('cool_girl',frame)
#     key = cv.waitKey(2)
#     if (key&0XFF) == ord('e'):
#         break
#     elif (key&0XFF) == ord('s'):
#         viwr.write(frame)
# cap.release()
# cv.destroyAllWindows()

#2，标记和numpy
# import cv2 as cv
# import numpy as np
# #窗口和读取
# cv.namedWindow('img1',0)
# cv.namedWindow('img2',0)
# cv.resizeWindow('img1',480,640)
# img1 = cv.imread("cool_girl.jpg")
# #numpy的使用
# img2 = np.zeros((640,480,3),np.uint8)
# while True:
#     key = cv.waitKey(1)
#     cv.imshow('img1', img1)
#     cv.imshow('img2', img2)
#     font = cv.FONT_HERSHEY_DUPLEX # 字体
#     cv.putText(img1,"cool girl",(150,80),font,3,(0,180,0),2,cv.LINE_AA)#img text pos font scale size color
#     cv.line(img2,(0,0),(200,200),(0,0,200),2)#图片，开始坐标，结束坐标，颜色（bgr），线条粗细。
#     if (key&0XFF) == ord('e'):
#         break
# cv.destroyAllWindows()

#3,ROI,拷贝,(b,g,r),draw,add,subtract,addWeighted
# import cv2 as cv
# import numpy as np
# #窗口和读取
# cv.namedWindow('img1',0)
# cv.namedWindow('img2',0)
# cv.namedWindow('img3',0)
# cv.resizeWindow('img2',480,640)
# cv.resizeWindow('img3',480,640)
# cv.resizeWindow('img1',480,640)
# img1 = cv.imread("cool_girl.jpg")
#
# #深拷贝
# img2 = img1.copy()
# #浅拷贝
# img3 = img1
# #拆分通道->浅拷贝
# b,g,r = cv.split(img3)
# b[10:100,10:100] = 0
# #合并通道->浅拷贝
# img4 = cv.merge((b,g,r))
# #画图->深拷贝
# cirle1 = cv.circle(img2,(100,200),10,(0,255,0),5,4)
# #ROI: y,x,ch
# img2[10:100,10:100,1] = 0
# #权值合并
# img5 = np.zeros((1280, 720, 3),np.uint8)
# img6 = cv.addWeighted(img1,0.8,img5,0.2,0)
# #print(img1.shape)
#
# cv.imshow('img1', img4)
# cv.imshow('img2', img2)
# cv.imshow('img3', img6)
# key = cv.waitKey(0)
# cv.destroyAllWindows()

#4,灰度变换,平移变换,加logo(关键点-灰度值相加)
# import cv2 as cv
# import numpy as np
# #加载图片
# img1 = cv.imread('cool_girl.jpg')
# img2 = cv.imread('girl.jpg')
# h1,w1,ch1 = img1.shape
# h2,w2,ch2 = img2.shape
# cv.namedWindow('img1',0)
# cv.namedWindow('img2',0)
# cv.namedWindow('img3',0)
# cv.resizeWindow('img1',480,640)
# cv.resizeWindow('img2',480,640)
# cv.resizeWindow('img3',480,640)
# # print(' h1=\n',h1,' w1=',w1,' ch1=',ch1,)#h1=1280  w1= 720  ch1= 3
# # print('\n h2=',h2,' w2=',w2,' ch2=',ch2,)#h2= 1000  w2= 1000  ch2= 3
# #灰度转换
# img1_gray = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
# img2_gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
# #平移矩阵(单位阵和平移量)
# M1 = cv.getRotationMatrix2D((w1/2,h1/2),90,1)
# M2 = np.float32([[1,0,100],[0,1,200]])
# img1_move = cv.warpAffine(img1_gray,M1,(w1,h1))
# img2_move = cv.warpAffine(img2_gray,M2,(w2,h2))
# #加logo:选logo,找logo显示位置,
# logo = img1[150:500,150:500]#(350, 350, 3)
# mask = np.zeros((350,350),np.uint8)#(350,350)
# mask[0:200,0:200] = 255#显示这个区域
#
# m = cv.bitwise_not(mask)#logo部分变黑
# roi = img1[0:350,0:350]#显示位置
# result = cv.bitwise_and(roi,roi,mask=m)#(350, 350,3)   roi与m
# # #print(result.shape,logo.shape)
# dst = cv.add(result,logo)#灰度值相加
#
# cv.imshow('img1', logo)
# cv.imshow('img2', result)#到这依旧正常
# cv.imshow('img3', dst)
# key = cv.waitKey(0)
# cv.destroyAllWindows()



