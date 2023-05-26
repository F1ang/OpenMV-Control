"""
一、入门教程
二、人脸识别教程
三、官方教程
四、B站学习
"""

                            #-----一、入门教程------#
#---1,初识---#
#picture and window#
# import cv2
# import matplotlib
# import numpy
#
# img = cv2.imread('cool_girl.jpg', 1)#1-彩色图 0-灰度图
# #处理图片窗口
# cv2.namedWindow('liangzai',0)#1-不可改，0-可改
# cv2.resizeWindow('liangzai', 480, 670)
# cv2.moveWindow('liangzai',1,0)
# cv2.imshow('liangzai',img)
#
# cv2.waitKey(0)#等待按键
# cv2.destroyAllWindows()

# ---2,视频---#
# import cv2
# cap = cv2.VideoCapture(0)#从计算机上的第一个网络摄像头返回视频
# #录制:out输出通信协议，cap捕获帧
# fourcc = cv2.VideoWriter_fourcc(*'XVID')#编解码器
# out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))#输出信息
# #cap->encode->out
# #out.write->show->release
# while True:
#     ret, frame = cap.read()
#     #其中ret是布尔值，如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False。
#     #frame就是每一帧的图像，是个三维矩阵。
#     if ret==True:
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#灰度转换
#         out.write(frame)#录制
#         cv2.imshow('frame', gray)
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#             break
# cap.release()
# out.release()
# cv2.destroyAllWindows()

# ---3,标记---#
# import numpy as np
# import cv2
# img1=cv2.imread('cool_girl.jpg',cv2.IMREAD_COLOR)
# cv2.namedWindow('cool_girl',0)
# cv2.resizeWindow('cool_girl', 480, 670)
#
# #cv2.line(img1,(0,0),(800,1500),(0,0,0),15)#图片，开始坐标，结束坐标，颜色（bgr），线条粗细。
# # cv2.rectangle(img1,(200,120),(450,420),(0,0,255),10)#左上角坐标，右下角坐标，颜色和线条粗细
# # cv2.circle(img1,(320,260), 150, (255,0,0), 0)#图像/帧，圆心，半径，颜色和
# #
# # pts = np.array([[10,5],[200,300],[200,400],[100,400],[50,10]], np.int32)#多边形
# # pts = pts.reshape((-1,1,2))
# # cv2.polylines(img1, [pts], True, (0,255,255), 5)
#
# font = cv2.FONT_HERSHEY_SIMPLEX #字体
# cv2.putText(img1, 'HELLO OPENCV!',(100,80), font, 2, (0,0,0), 5, cv2.LINE_AA)
# #图片，文字，文字起位置，字体,文字大小，文字颜色，文字粗细，cv2.LINE_AA
#
# cv2.imshow('cool_girl',img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ---4,图像操作---#
# import numpy as np
# import cv2
# img1=cv2.imread('cool_girl.jpg',cv2.IMREAD_COLOR)
# cv2.namedWindow('cool_girl',0)
# cv2.resizeWindow('cool_girl', 480, 670)
# font = cv2.FONT_HERSHEY_SIMPLEX #字体
# cv2.putText(img1, 'HELLO OPENCV!',(100,80), font, 2, (0,0,0), 5, cv2.LINE_AA)
# #取像素,存像素,显像素(区域)
# img1[100:150,100:150] = [0,255,0]#rgb--取1
# #img1[250:324,20:107] = img1[37:111,107:194] #[y1:y2,x1:x2] x为纵向，y为横向 取2
#
# cv2.imshow('cool_girl',img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ---5,图像算术和逻辑运算---#
# import cv2
# import numpy as np
# img1 = cv2.imread('cool_girl.jpg')
# img2 = cv2.imread('cool_girl.jpg')
# cv2.namedWindow('A',0)
# cv2.resizeWindow('A', 480, 640)
# weighted = cv2.addWeighted(img1,0.7, img2, 0.3, 0) #权重 光线伽马值--add
# cv2.imshow('A',weighted)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #图像区域（ROI）,根据阈值将所有像素转换为黑色或白色->白色遮住->遮住部分换背景
# import cv2
# import numpy as np
# #Load two images
# img1 = cv2.imread('cool_girl.jpg')
# img2 = cv2.imread('cool_girl.jpg')
# cv2.namedWindow('A',0)
# #ROI
# rows, cols, channels = img2.shape
# roi = img1[0:rows, 0:cols]
# #create its inverse mask(灰度背景)
# img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
# #hreshold，去除低阈值.
# ret , mask = cv2.threshold(img2gray, 220, 255, cv2.THRESH_BINARY_INV)#阈值区logo
# mask_inv = cv2.bitwise_not(mask)#非阈值区logo
#
# #logo in ROI,位与操作，注:标注即遮住，阈值区与非阈值区--背景，logo
# img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)#标注logo的非阈值区(运算时全白,即遮住)
# #背景和logo位与(logo非阈值区)
# img2_fg = cv2.bitwise_and(img2, img2, mask=mask)#标注背景在logo阈值区
# dst = cv2.add(img1_bg,img2_fg)#再背景非阈值区和logo融合
#
# img1[0:rows, 0:cols] = dst
# cv2.waitKey(0)
# cv2.imshow('A',img1)
# cv2.destroyAllWindows()

# ---6,阈值---#
# import cv2
# import numpy as np
# img = cv2.imread('cool_girl.jpg')
# retval, threshold = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)#二元阈值(突黑显白)
# cv2.imshow('original',img)
# cv2.imshow('threshold',threshold)#----有颜色
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# img = cv2.imread('cool_girl.jpg')
# grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#灰度化图像
# retval, threshold = cv2.threshold(grayscaled, 125, 255, cv2.THRESH_BINARY)
# cv2.imshow('original',img)
# cv2.imshow('threshold',threshold)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# img = cv2.imread('cool_girl.jpg')
# grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# th = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 125, 1)#自适应阈值
# cv2.imshow('original',img)
# cv2.imshow('Adaptive threshold',th)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# #大津阈值
# #retval2,threshold2 = cv2.threshold(grayscaled,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)cv2.imshow(‘original’,img)

# ---7,颜色过滤---#
# import cv2
# import numpy as np
# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     lower_red = np.array([78,43,46])#找颜色
#     upper_red = np.array([99,255,255])
#     mask = cv2.inRange(hsv, lower_red, upper_red)#掩码ROI
#     res = cv2.bitwise_and(frame,frame, mask= mask)#按位与,标注掩码ROI的原图
#     cv2.imshow('frame',frame)
#     cv2.imshow('mask',mask)
#     cv2.imshow('res',res)
#     if cv2.waitKey(5) & 0xFF == ord('q'):
#         break
# cv2.destroyAllWindows()
# cap.release()

# ---8,模糊和平滑(滤波)---#
# import cv2
# import numpy as np
# cap = cv2.VideoCapture(0)
# while(1):
#  ret, frame = cap.read()
#  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#  lower_red = np.array([30,150,50])
#  upper_red = np.array([255,255,180])
#  mask = cv2.inRange(hsv, lower_red, upper_red)
#  res = cv2.bitwise_and(frame,frame, mask= mask)#取颜色图(标注原图的阈值区)
#  #均值滤波
#  kernel = np.ones((15, 15), np.float32) / 225   #225个像素均值,核15*15
#  smoothed = cv2.filter2D(res, -1, kernel)       #均值滤波图
#  #高斯模糊
#  blur = cv2.GaussianBlur(res,(15,15),0)
#  #中值滤波
#  median = cv2.medianBlur(res,15)
#  #双向滤波
#  bilateral = cv2.bilateralFilter(res, 15, 75, 75)
#
#  cv2.imshow('bilateralBlur', bilateral)
#  cv2.imshow('Median Blur',median)
#  cv2.imshow('Gaussian Blurring',blur)
#  cv2.imshow('Original', frame)
#  cv2.imshow('Averaging', smoothed)
#  k = cv2.waitKey(5) & 0xFF#Esc
#  if k == 27:
#      break
# cv2.destroyAllWindows()
# cap.release()

# ---9,形态变换---#
# #腐蚀与膨胀:不腐白，不膨黑
# import cv2
# import numpy as np
# cap = cv2.VideoCapture(0)
# while(1):
#  res, frame = cap.read()
#  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#  lower_red = np.array([30,150,50])
#  upper_red = np.array([255,255,180])
#  mask = cv2.inRange(hsv, lower_red, upper_red)#掩码ROI
#  res = cv2.bitwise_and(frame,frame, mask= mask)#在原图取颜色图(即标注掩码区的原图)
#
#  kernel = np.ones((5,5),np.uint8)#滑块（核）:5*5
#  erosion = cv2.erode(mask,kernel,iterations = 1)#腐蚀
#  dilation = cv2.dilate(mask,kernel,iterations = 1)#膨胀      不腐白，不膨黑(对掩码ROI)
#  cv2.imshow('Original',frame)
#  cv2.imshow('Mask',mask)
#  cv2.imshow('Erosion',erosion)
#  cv2.imshow('Dilation',dilation)
#  k = cv2.waitKey(5) & 0xFF
#  if k == 27:
#   break
# cv2.destroyAllWindows()
# cap.release()

#开运算与闭运算:腐蚀和膨胀先后进行
# import cv2
# import numpy as np
# cap = cv2.VideoCapture(0)
# while(1):
#  res, frame = cap.read()
#  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#  lower_red = np.array([30,150,50])
#  upper_red = np.array([255,255,180])
#  mask = cv2.inRange(hsv, lower_red, upper_red)
#  res = cv2.bitwise_and(frame,frame, mask= mask)
#  kernel = np.ones((5,5),np.uint8)
#
#  opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)#开运算
#  closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)#闭运算---背景清黑像素
#
#  cv2.imshow('Original',frame)
#  cv2.imshow('Mask',mask)
#  cv2.imshow('Opening',opening)
#  cv2.imshow('Closing',closing)
#  k = cv2.waitKey(5) & 0xFF
#  if k == 27:
#    break
# cv2.destroyAllWindows()
# cap.release()

# ---10,边缘检测和渐变---#
#Sobel算子
# import cv2
# import numpy as np
# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     lower_red = np.array([30,150,50])
#     upper_red = np.array([255,255,180])
#     mask = cv2.inRange(hsv, lower_red, upper_red)
#     res = cv2.bitwise_and(frame,frame, mask= mask)#标注掩码区的原图
#
#     laplacian = cv2.Laplacian(frame,cv2.CV_64F)
#     sobelx = cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=5)#Sobel算子:梯度与灰度变化,x方向
#     sobely = cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=5)
#
#     cv2.imshow('frame', frame)
#     cv2.imshow('mask', mask)
#     cv2.imshow('laplacian',laplacian)
#     cv2.imshow('sobelx',sobelx)
#     cv2.imshow('sobely',sobely)
#     if cv2.waitKey(5) & 0xFF == ord('q'):
#         break
# cv2.destroyAllWindows()
# cap.release()

# #Canny 边缘检测算子
# import cv2
# import numpy as np
# cap = cv2.VideoCapture(0)
# while True:
#     _, frame = cap.read()
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     lower_red = np.array([30,150,50])
#     upper_red = np.array([255,255,180])
#     mask = cv2.inRange(hsv, lower_red, upper_red)
#     res = cv2.bitwise_and(frame,frame, mask= mask)
#
#     cv2.imshow('Original',frame)
#     edges = cv2.Canny(frame,100,200)#Canny 边缘检测算子
#     cv2.imshow('Edges',edges)
#     k = cv2.waitKey(5) & 0xFF
#     if k == 27:
#         break
# cv2.destroyAllWindows()
# cap.release()


# ---11,模板匹配---#
# import cv2
# import numpy as np
# img = cv2.imread('cool_girl.jpg', 0)
# template = cv2.imread('cool_girl_face.jpg', 0)
# cv2.namedWindow('Edges',0)
# cv2.resizeWindow('Edges', 480, 670)
# h, w = template.shape[:2]   #rows->h, cols->w,template size
# #相关系数匹配方法：cv2.TM_CCOEFF
# res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)#灰度图(最白的地方表示最大的匹配)
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)#最大匹配值的坐标
#
# left_top = max_loc  # 左上角
# right_bottom = (left_top[0] + w, left_top[1] + h)  #右下角
# cv2.rectangle(img, left_top, right_bottom, (0, 255, 0), 2)  # 画出矩形位置
#
# cv2.imshow('Edges',img)
# cv2.waitKey(0)

# 阈值匹配(可进行多次模板匹配)
#1.读入原图和模板
# import cv2
# import numpy as np
# img_rgb = cv2.imread('cool_girl.jpg')
# img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
# template = cv2.imread('cool_girl_face.jpg', 0)
# h, w = template.shape[:2]
# #2.标准相关模板匹配
# res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
# threshold = 0.8                 #阈值
# loc = np.where(res >= threshold)  #匹配程度大于%80的坐标y,x
# for pt in zip(*loc[::-1]):         #翻转得x,y->拼接
#     right_bottom = (pt[0] + w, pt[1] + h)
#     cv2.rectangle(img_rgb, pt, right_bottom, (0, 0, 255), 1)
# cv2.imshow('Detected',img_rgb)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# ---12,GrabCut 前景提取---#
# import numpy as np       #数组
# import cv2
# from matplotlib import pyplot as plt
# img = cv2.imread('cool_girl.jpg')
# mask = np.zeros(img.shape[:2],np.uint8)#img掩码数组
# bgdModel = np.zeros((1,65),np.float64)#background
# fgdModel = np.zeros((1,65),np.float64)#forground,容器
# rect = (170,150,250,250)#提取范围
#
# cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)#5次迭代，前景图
# mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')#背景图
# img = img*mask2[:, :, np.newaxis]
# plt.imshow(img)
# plt.colorbar()
# plt.show()
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# ---13,角点检测---#
# import numpy as np
# import cv2
# cv2.namedWindow('Corner',0)
# cv2.resizeWindow('Corner', 480, 670)
# img = cv2.imread('cool_girl.jpg')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# gray = np.float32(gray)
# corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)#img max_num quilt min_dis
# corners = np.int0(corners)#limit
#
# for corner in corners:   #在角点处画圆
# 	x,y = corner.ravel() #角点数据解析
# 	cv2.circle(img,(x,y),3,(0,0,150),-1)
# cv2.imshow('Corner',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# ---14,特征匹配（单映射）爆破---#
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# img1 = cv2.imread('cool_girl.jpg',0)
# img2 = cv2.imread('cool_girl_face.jpg',0)
#
# orb = cv2.ORB_create()                      #特征提取器
# kp1, des1 = orb.detectAndCompute(img1,None) #特征，描述符
# kp2, des2 = orb.detectAndCompute(img2,None)
#
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = bf.match(des1,des2)               #特征匹配
# matches = sorted(matches, key = lambda x:x.distance)    #距离排序
# img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)#前十个匹配
# plt.imshow(img3)
# plt.show()
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# ---15,MOG 背景减弱(削静态)---#
# import numpy as np
# import cv2
# cap = cv2.VideoCapture(0)
# fgbg = cv2.createBackgroundSubtractorMOG2()
# while(1):
# 	ret, frame = cap.read()
# 	fgmask = fgbg.apply(frame)
# 	cv2.imshow('fgmask',frame)
# 	cv2.imshow('frame',fgmask)
# 	k = cv2.waitKey(30) & 0xff
# 	if k == 27:
# 		break
# cap.release()
# cv2.destroyAllWindows()


# ---16,Haar Cascade(层叠)面部检测---#
# import numpy as np
# import cv2
# face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
# cap = cv2.VideoCapture(0)
# while True:
# 	ret, img = cap.read()
# 	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 	faces = face_cascade.detectMultiScale(gray, 1.3, 5)#找到脸部
# 	for (x,y,w,h) in faces:		#拆分脸部
# 		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
# 		roi_gray = gray[y:y+h, x:x+w]
# 		roi_color = img[y:y+h, x:x+w]
# 		eyes = eye_cascade.detectMultiScale(roi_gray)#找到eyes
# 		for (ex,ey,ew,eh) in eyes:
# 			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
# 	cv2.imshow('img',img)
# 	k = cv2.waitKey(30) & 0xff
# 	if k == 27:
# 		break
# cap.release()
# cv2.destroyAllWindows()


# ---17,创建自己的 Haar Cascade & 训练---#
# import numpy as np
# import cv2
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#训练集
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')#this is the cascade we just made. Call what you want
# watch_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
# cap = cv2.VideoCapture(0)
# while 1:
#     ret, img = cap.read()
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # add this # image, reject levels level weights.
#     watches = watch_cascade.detectMultiScale(gray, 50, 50)  # add this for (x,y,w,h) in watches:
#     for (x,y,w,h) in faces:
#         cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = img[y:y+h, x:x+w]
#         eyes = eye_cascade.detectMultiScale(roi_gray)
#         for (ex,ey,ew,eh) in eyes:
#             cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
#     cv2.imshow('img',img)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break
# cap.release()
# cv2.destroyAllWindows()


                        #-----二、人脸识别教程------#
# import cv2
# import numpy as np
# 封装函数,展示图片
# def cv_show(name,img):
#     cv2.imshow(name,img)
#     key = cv2.waitKey(0)
#     if key & 0xFF == ord('q'):
#         cv2.destroyAllWindows()

# # 创建纯黑的背景来画图
# img = np.zeros((480,640,3),np.uint8)
# # 画线
# cv2.line(img, (10,20),(300,400), (0,0,255), 5)
# # 画矩形
# cv2.rectangle(img, (10,20),(300,400), (0,0,255), 5)
# # 画圆 参数：圆心坐标，半径大小，颜色，粗细
# cv2.circle(img, (320,240),100,(0,0,255), 5)
# # 画文字
# cv2.putText(img,'Hello OpenCV',(50,400),cv2.FONT_HERSHEY_COMPLEX,2,[0,0,255])
# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # 打开摄像头
# cap = cv2.VideoCapture(0) # cap代表被打开的摄像头
# while 1:
#     # 从摄像头中读取一帧数据
#     ret, frame = cap.read() # frame 存储了读取到的数据
#     cv2.imshow('img',frame)
#     key = cv2.waitKey(1000//24)
#     if key == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()


##摄像头人脸识别
##使用opencv提供的人脸识别算法(人脸识别器)
# import cv2
# import numpy as np
# face_detector = cv2.CascadeClassifier('D:\Python-File-C++\picture\haarcascades\haarcascade_frontalface_alt.xml')
# # 打开摄像头
# cap = cv2.VideoCapture(0)  # cap代表被打开的摄像头
# while 1:
#     # 从摄像头中读取一帧数据
#     ret, frame = cap.read()  # frame 存储了读取到的数据
#     # 将获取到的数据进行灰度处理
#     gray = cv2.cvtColor(frame, code=cv2.COLOR_BGR2GRAY)
#     # 在进行灰度处理后的图片中识别人脸
#     faces = face_detector.detectMultiScale(gray)  #被识别到的多张人脸!!
#     print('没检测到人脸\n')
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)#BGR
#         print('检测到人脸\n')
#     cv2.imshow('frame', frame)
#     key = cv2.waitKey(1000 // 24)
#     if key == ord('q'):
#         break
#     # cv2.imwrite('tong.jpg',frame)
# cap.release()
# cv2.destroyAllWindows()

# import cv2 as cv
# import numpy as np
# img=cv.imread('cool_girl.jpg',1)
# cv.namedWindow('cool_girl',0)
# cv.resizeWindow('cool_girl',480,680)
# cv.imshow('cool_girl',img)
# cv.waitKey(0)
# cv.destroyAllWindows()
# #picture attribute


                            #-----三、官方教程------#
#1、图像入门
#1)show picture
# import numpy as np
# import cv2 as cv
# #加载彩色灰度图像
# img = cv.imread('cool_girl.jpg',cv.IMREAD_COLOR)#IMREAD_COLOR=1
# cv.namedWindow('cool_girl',cv.WINDOW_NORMAL)#WINDOW_NORMAL=1
# cv.imshow('cool_girl',img)
# key=cv.waitKey(0)
# if key==ord('s'):
#     cv.imwrite('cool_girl.jpg',img)
#     cv.destroyAllWindows()
# elif key==27:#esc
#     cv.destroyAllWindows()

#2)Matplotlib绘图库
# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
# img = cv.imread('cool_girl.jpg',0)
# plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
# #plt.xticks([]), plt.yticks([]) # 隐藏 x 轴和 y 轴上的刻度值
# plt.show()

#3)绘图
# import numpy as np
# import cv2 as cv
# # 创建黑色的图像
# img = np.zeros((512,512,3), np.uint8)
# # 绘制一条厚度为5的蓝色对角线
# cv.line(img,(0,0),(511,511),(255,0,0),5)
# font = cv.FONT_HERSHEY_SIMPLEX
# cv.putText(img,'OpenCV',(70,90), font, 3,(255,255,255),2,cv.LINE_AA)
#
# cv.imshow('cool_girl',img)
# key=cv.waitKey(0)
# if key==ord('s'):
#     cv.imwrite('cool_girl.jpg',img)
#     cv.destroyAllWindows()
# elif key==27:#esc
#     cv.destroyAllWindows()

#事件回调
# import numpy as np
# import cv2 as cv
#
# drawing = False # 如果按下鼠标，则为真
# mode = True # 如果为真，绘制矩形。按 m 键可以切换到曲线
# ix,iy = -1,-1
# img = np.zeros((512,512,3), np.uint8)
# # 鼠标回调函数(事件回调)
# def draw_circle(event,x,y,flags,param):
#  global ix,iy,drawing,mode
#  if event == cv.EVENT_LBUTTONDOWN:
#     drawing = True
#     ix,iy = x,y
#  elif event == cv.EVENT_MOUSEMOVE:
#     if drawing == True:
#         if mode == True:
#          cv.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
#     else:
#          cv.circle(img,(x,y),5,(0,0,255),-1)
#  elif event == cv.EVENT_LBUTTONUP:
#     drawing = False
#     if mode == True:
#         cv.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
#  else:
#     cv.circle(img,(x,y),5,(0,0,255),-1)
#
# cv.namedWindow('image')
# cv.setMouseCallback('image',draw_circle)
# while(1):
#  cv.imshow('image',img)
#  if cv.waitKey(20) & 0xFF == 27:
#     break
# cv.destroyAllWindows()

#事件轨迹及属性获取
# import numpy as np
# import cv2 as cv
# img = np.zeros((300,512,3), np.uint8)
# cv.namedWindow('image')
#
# #事件绑定
# def draw_b():
#     pass
# def draw_g():
#     pass
# def draw_r():
#     pass
# cv.createTrackbar('B','image',0,255,draw_b)
# cv.createTrackbar('G','image',0,255,draw_g)
# cv.createTrackbar('R','image',0,255,draw_r)
#
# while(1):
#  cv.imshow('image',img)
#  k = cv.waitKey(1) & 0xFF
#  if k == 27:
#     break
#  #获取事件属性
#  r = cv.getTrackbarPos('R','image')
#  g = cv.getTrackbarPos('G','image')
#  b = cv.getTrackbarPos('B','image')
#  img[:] = [b,g,r]
# cv.destroyAllWindows()


#4)图像运算,性能,颜色空间(!!),几何变换

#对蓝色对象的跟踪
# import cv2 as cv
# import numpy as np
# cap = cv.VideoCapture(0)
# while(1):
#  # 读取帧
#  _, frame = cap.read()
#  # 转换颜色空间 BGR 到 HSV
#  hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
#  # 定义HSV中蓝色的范围
#  lower_blue = np.array([110,50,50])
#  upper_blue = np.array([130,255,255])
#  # 设置HSV的阈值使得只取蓝色
#  mask = cv.inRange(hsv, lower_blue, upper_blue)
#  # 将掩膜和图像逐像素相加
#  res = cv.bitwise_and(frame,frame, mask= mask)#scr1 scr2,dst=dst(I)=src1(I)∧src2 if mask(I)≠0,mask
#  cv.imshow('frame',frame)
#  cv.imshow('mask',mask)
#  cv.imshow('res',res)
#  k = cv.waitKey(5) & 0xFF
#  if k == 27:
#     break
# cv.destroyAllWindows()

#5)图像阈值(小0大取最大),图像平滑,形态学转换,图像梯度(边缘),Canny边缘检测

#1)图像梯度
# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
# img = cv.imread('cool_girl.jpg',0)#读入灰度图
# laplacian = cv.Laplacian(img,cv.CV_64F)
# sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)#ddepth=-1
# sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)
# plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
# plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
# plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
# plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
# plt.show()

#2)Canny边缘检测
# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
# img = cv.imread('cool_girl.jpg',0)
# edges = cv.Canny(img,100,200)#Canny梯度算子
# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()

#6、图像金字塔
# import cv2 as cv
# import numpy as np,sys
# A = cv.imread('cool_girl.jpg')
# B = cv.imread('girl.jpg')
# # 生成A的高斯金字塔
# G = A.copy()
# gpA = [G]
# xrange(6)=(0,1,2,3,4,5)
# for i in xrange(6):
#  G = cv.pyrDown(G)
#  gpA.append(G)
# # 生成B的高斯金字塔
# G = B.copy()
# gpB = [G]
# for i in xrange(6):
#  G = cv.pyrDown(G)
#  gpB.append(G)
# # 生成A的拉普拉斯金字塔
# lpA = [gpA[5]]
# for i in xrange(5,0,-1):
#  GE = cv.pyrUp(gpA[i])
#  L = cv.subtract(gpA[i-1],GE)
#  lpA.append(L)
# # 生成B的拉普拉斯金字塔
# lpB = [gpB[5]]
# for i in xrange(5,0,-1):
#  GE = cv.pyrUp(gpB[i])
#  L = cv.subtract(gpB[i-1],GE)
#  lpB.append(L)
# # 现在在每个级别中添加左右两半图像
# LS = []
# for la,lb in zip(lpA,lpB):
#  rows,cols,dpt = la.shape
#  ls = np.hstack((la[:,0:cols/2], lb[:,cols/2:]))
#  LS.append(ls)
# # 现在重建
# ls_ = LS[0]
# for i in xrange(1,6):
#  ls_ = cv.pyrUp(ls_)
#  ls_ = cv.add(ls_, LS[i])
# # 图像与直接连接的每一半
# real = np.hstack((A[:,:cols/2],B[:,cols/2:]))
# cv.imshow('Pyramid_blending2.jpg',ls_)
# cv.imshow('Direct_blending.jpg',real)
# cv.waitKey(0)
# cv.destroyAllWindows()

#7、轮廓
# import cv2 as cv
# import numpy as np
# img1 = cv.imread('cool_girl.jpg',0)
# img2 = cv.imread('cool_girl_face.jpg',0)
# ret, thresh = cv.threshold(img1, 127, 255,0)
# ret, thresh2 = cv.threshold(img2, 127, 255,0)
# contours,hierarchy = cv.findContours(thresh,2,1)
# cnt1 = contours[0]
# contours,hierarchy = cv.findContours(thresh2,2,1)
# cnt2 = contours[0]
# ret = cv.matchShapes(cnt1,cnt2,1,0.0)
# print( ret )

#8、直方图

# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
# img = cv.imread('cool_girl.jpg')
# hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
# hist = cv.calcHist( [hsv], [0, 1], None, [180, 256], [0, 180, 0, 256] )
# plt.imshow(hist,interpolation = 'nearest')
# plt.show()

# import numpy as np
# import cv2 as cv
# roi = cv.imread('cool_girl.jpg')
# hsv = cv.cvtColor(roi,cv.COLOR_BGR2HSV)
# target = cv.imread('cool_girl_face.jpg')
# hsvt = cv.cvtColor(target,cv.COLOR_BGR2HSV)
# # 计算对象的直方图
# roihist = cv.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
# # 直方图归一化并利用反传算法
# cv.normalize(roihist,roihist,0,255,cv.NORM_MINMAX)
# dst = cv.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)
# # 用圆盘进行卷积
# disc = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
# cv.filter2D(dst,-1,disc,dst)
# # 应用阈值作与操作
# ret,thresh = cv.threshold(dst,50,255,0)
# thresh = cv.merge((thresh,thresh,thresh))
# res = cv.bitwise_and(target,thresh)
# res = np.vstack((target,thresh,res))
# cv.imshow('cool',res)
# cv.waitKey(0)
# cv.destroyAllWindows()


#9、模板匹配
# import cv2 as cv
# import numpy as np
# from matplotlib import pyplot as plt
# cv.namedWindow('cool',0)
# cv.resizeWindow('cool',640,480)
# img_rgb = cv.imread('cool_girl.jpg')
# img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
# template = cv.imread('cool_girl_face.jpg',0)
# w, h = template.shape[::-1]
# res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
# threshold = 0.8
# loc = np.where( res >= threshold)
# for pt in zip(*loc[::-1]):
#  cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 1,3)
#  cv.imshow('cool',img_rgb)
# cv.waitKey(0)
# cv.destroyAllWindows()

#10、霍夫曼变换
# import cv2 as cv
# import numpy as np
# cv.namedWindow('houghlines3',0)
# cv.resizeWindow('houghlines3',480,640)
# img = cv.imread(cv.samples.findFile('cool_girl.jpg'))
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# edges = cv.Canny(gray,50,150,apertureSize = 3)
# lines = cv.HoughLines(edges,1,np.pi/180,200)
# for line in lines:
#  rho,theta = line[0]
#  a = np.cos(theta)
#  b = np.sin(theta)
#  x0 = a*rho
#  y0 = b*rho
#  x1 = int(x0 + 1000*(-b))
#  y1 = int(y0 + 1000*(a))
#  x2 = int(x0 - 1000*(-b))
#  y2 = int(y0 - 1000*(a))
#  cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)
# cv.imshow('houghlines3',img)
# cv.waitKey(0)
# cv.destroyAllWindows()

#霍夫圈变换
# import numpy as np
# import cv2 as cv
# img = cv.imread('cool_girl.jpg',0)
# img = cv.medianBlur(img,5)
# cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
# circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,20,
#  param1=50,param2=30,minRadius=0,maxRadius=0)
# circles = np.uint16(np.around(circles))
# for i in circles[0,:]:
#  # 绘制外圆
#  cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
#  # 绘制圆心
#  cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
# cv.imshow('detected circles',cimg)
# cv.waitKey(0)
# cv.destroyAllWindows()

#11、特征:哈里斯角检测及其改进算法

#1)哈里斯角检测
# import numpy as np
# import cv2 as cv
# img = cv.imread('cool_girl.jpg')
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# gray = np.float32(gray)
# dst = cv.cornerHarris(gray,2,3,0.04)
# #result用于标记角点，并不重要
# dst = cv.dilate(dst,None)
# #最佳值的阈值，它可能因图像而异。
# img[dst>0.01*dst.max()]=[0,0,255]
# cv.imshow('dst',img)
# if cv.waitKey(0) & 0xff == 27:
#  cv.destroyAllWindows()

#2)改进
# import numpy as np
# import cv2 as cv
# cv.namedWindow('corner',0)
# cv.resizeWindow('corner',480,640)
# img = cv.imread('cool_girl.jpg')
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# # 寻找哈里斯角
# gray = np.float32(gray)
# dst = cv.cornerHarris(gray,2,3,0.04)
# dst = cv.dilate(dst,None)
# ret, dst = cv.threshold(dst,0.01*dst.max(),255,0)
# dst = np.uint8(dst)
# # 寻找质心
# ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
# # 定义停止和完善拐角的条件
# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
# corners = cv.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
# # 绘制
# res = np.hstack((centroids,corners))
# res = np.int0(res)
# img[res[:,1],res[:,0]]=[0,0,255]
# img[res[:,3],res[:,2]] = [0,255,0]
# cv.imshow('corner',img)
# cv.waitKey(0)
# cv.destroyAllWindows()

#3)关键点和描述符
# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
# img = cv.imread('cool_girl.jpg',0)
# # 初始化FAST检测器
# star = cv.xfeatures2d.StarDetector_create()
# # 初始化BRIEF提取器
# brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
#
# # 找到STAR的关键点
# kp = star.detect(img,None)
# # 计算BRIEF的描述符
# kp, des = brief.compute(img, kp)
# print( brief.descriptorSize() )
# print( des.shape )

#4)
# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
# img = cv.imread('cool_girl.jpg',0)
# # 初始化ORB检测器
# orb = cv.ORB_create()
# # 用ORB寻找关键点
# kp = orb.detect(img,None)
# # 用ORB计算描述符
# kp, des = orb.compute(img, kp)
# # 仅绘制关键点的位置，而不绘制大小和方向
# img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
# plt.imshow(img2), plt.show()


#5)特征匹配
# import numpy as np
# import cv2 as cv
# import matplotlib.pyplot as plt
# img1 = cv.imread('cool_girl.jpg',cv.IMREAD_GRAYSCALE) # 索引图像
# img2 = cv.imread('cool_girl_face.jpg',cv.IMREAD_GRAYSCALE) # 训练图像
# # 初始化SIFT描述符
# sift = cv.xfeatures2d.SIFT_create()
# # 基于SIFT找到关键点和描述符
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)
# # FLANN的参数
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks=50) # 或传递一个空字典
# flann = cv.FlannBasedMatcher(index_params,search_params)
# matches = flann.knnMatch(des1,des2,k=2)
# # 只需要绘制好匹配项，因此创建一个掩码
# matchesMask = [[0,0] for i in range(len(matches))]
# # 根据Lowe的论文进行比例测试
# for i,(m,n) in enumerate(matches):
#  if m.distance < 0.7*n.distance:
#   matchesMask[i]=[1,0]
#   draw_params = dict(matchColor = (0,255,0),
#   singlePointColor = (255,0,0),
#   matchesMask = matchesMask,
#   flags = cv.DrawMatchesFlags_DEFAULT)
# img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
# plt.imshow(img3,),plt.show()


#12、背景分离(静态背景分离出动态)
#1)分离
# from __future__ import print_function
# import cv2 as cv
# import argparse
# parser = argparse.ArgumentParser(description='This program shows how to use\
# background subtraction methods provided by OpenCV. You can process both videos and images.')
# parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='engineer2022.mp4')
# parser.add_argument('--algo', type=str, help='Background subtraction method (KNN,MOG2).', default='MOG2')
# args = parser.parse_args()
# if args.algo == 'MOG2':
#  backSub = cv.createBackgroundSubtractorMOG2()
# else:
#  backSub = cv.createBackgroundSubtractorKNN()
# capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
# if not capture.isOpened:
#  print('Unable to open: ' + args.input)
#  exit(0)
# while True:
#  ret, frame = capture.read()
#  if frame is None:
#   break
#  fgMask = backSub.apply(frame)
#  cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
#  cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
#  cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
#  cv.imshow('Frame', frame)
#  cv.imshow('FG Mask', fgMask)
#  keyboard = cv.waitKey(30)
#  if keyboard == 'q' or keyboard == 27:
#   break

#2)分离追踪(改进)
#Meanshift和Camshift

#13、级联分类器


#四、OpenCV图像处理(视频)

#1、








