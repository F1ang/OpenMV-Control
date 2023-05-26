#1,
# import cv2 as cv  
# import numpy as np
# cv.namedWindow('309',0)
# cv.resizeWindow('309',640,480)
# img=cv.imread('D:/Python-File-C++/picture/309_boys.jpg')
# while True:
#     cv.imshow('309',img)
#     key=cv.waitKey(0)
#     if(key & 0xFF== ord('q')):#ascii为8位
#          break    
#     elif(key & 0xff==ord('s')):
#         cv.imwrite('309_boys_cope.png',img)
#     else:
#         print(key)    
# cv.destroyAllWindows()

# cap=cv.VideoCapture(0,0)
# ret,frame=cap.read()
# cap.release()
# fourcc = cv.VideoWriter_fourcc()
# cv.VideoWriter(path,fourcc,25,(640,480))


#2,
# from tkinter.tix import ExFileSelectBox
# import cv2 as cv
# #创建VideoWriter位写多媒体文件(写入的格式)
# fourcc = cv.VideoWriter_fourcc(*'MJPG')
# viwr = cv.VideoWriter('./out1.mp4',fourcc,24,(640,480))#output 编码器 帧率 分辨率
# #创建窗口
# cv.namedWindow('f1ang',cv.WINDOW_NORMAL)
# cv.resizeWindow('f1ang',640,480)
# #获取视频设备/从视频文件获取视频帧
# cap = cv.VideoCapture(0)
# #判断摄像头是否打开
# while cap.isOpened():
#     #从摄像头获取帧
#     ret,frame = cap.read()
#     if ret == True:
#         cv.resizeWindow('f1ang',640,480)
#         cv.imshow('f1ang',frame)
#         #写入多媒体文件
#         viwr.write(frame)
#         #1ms
#         key = cv.waitKey(1)
#         if(key & 0xFF == ord('q')):
#             break
#     else:
#          break
# #释放
# cap.release()
# viwr.release()
# cv.destroyAllWindows


#3,
# import cv2   as cv
# import numpy as np

# a = np.array([1,2,3])
# b = np.array([[1,2,3],[4,5,6]])
# c = np.zeros((4,4,3),np.uint8)
# d = np.ones((8,8),np.uint8)
# e = np.full((4,5,3),7,np.uint8)
# f = np.identity(4)
# g = np.eye(3,4,k=2)

# print(a,b,c,d,e,f,g)

#4,
# import cv2 as cv
# from cv2 import polylines
# import numpy as np
# #ROI
# img = np.zeros((480,640,3),np.uint8)
# b,g,r = cv.split(img)
# b[10:100,10:100] = 250
# g[10:100,10:100] = 100
# img2 = cv.merge((b,g,r))
# #draw
# img3 = img.copy()#深拷贝
# line1 = cv.line(img3,(0,0),(100,100),(0,0,255),5,4)
# cirle1 = cv.circle(img3,(100,200),5,(0,255,0),5,4)
# ellipsise1 = cv.ellipse(img3,(500,100),(100,50),0,0,300,(255,0,0),5,4)
# pts = np.array([(300,10),(150,100),(450,100)],np.int32)#等腰三角形
# polylines1 = cv.polylines(img3,[pts],True,(0,0,255))
# cv.fillPoly(polylines1,[pts],(255,0,0))#img的roi区域
# cv.putText(img3,'hello opencv',(150,200),1,3,(0,255,0),5,4)
# #show
# cv.imshow('img',img)
# cv.imshow('b',b)
# cv.imshow('g',g)
# cv.imshow('img2',img2)

# cv.imshow('line1',line1)
# cv.imshow('crile1',cirle1)
# cv.imshow('ellipse1',ellipsise1)
# cv.imshow('polylines1',polylines1)
# cv.waitKey(0)
# cv.destroyAllWindows()

#5,
# import cv2 as cv
# import numpy as np

# img1 = cv.imread('309_boys.jpg')
# print(img1.shape)
# img2 = np.ones((640,963,3),np.uint8)*50#曝光度
# result = cv.add(img1,img2)
# result1 = cv.subtract(result,img2)
# result2 = cv.addWeighted(img1,0.3,img2,0.7,0)

# cv.imshow('result',result)
# cv.imshow('result1',result1)
# cv.imshow('result2',result2)
# cv.imshow('img1',img1)
# cv.waitKey(0)
# cv.destroyAllWindows()


#6,
# import cv2 as cv
# import numpy as np
# from scipy.fft import dst

# img1 = cv.imread('309_boys.jpg')
# h,w,ch = img1.shape
# #平移矩阵(单位阵和平移量)
# #M = np.float32([[1,0,100],[0,1,200]])#way1 前左右，后上下(正或者负)

# #M = cv.getRotationMatrix2D((w/2,h/2),40,0.3)#way2

# src = np.float32([[100,100],[200,100],[300,300]])#way3  三个坐标点
# dst= np.float32([[100,200],[200,200],[300,400]])
# M = cv.getAffineTransform(src,dst)

# #平移变换
# img2 = cv.warpAffine(img1,M,(w,h))

# cv.imshow('img1',img1)
# cv.imshow('img2',img2)
# cv.waitKey(0)
# cv.destroyAllWindows()

#7,
# import cv2 as cv
# import numpy as np
# src = cv.imread('309_boys.jpg')

# kernel = np.ones((5,5),np.float32)/25#削峰值，达平滑
# dst=cv.filter2D(src,-1,kernel)

# #均值滤波
# dst = cv.blur(src,(5,5))
# #高斯滤波
# dst = cv.GaussianBlur(src,(5,5),sigmaX=1)
# #中值滤波
# dst = cv.medianBlur(src,5)
# #双边滤波
# dst = cv.bilateralFilter(src,7,20,20)
# #----------------
# #Sobel(边缘)
# dst = cv.Sobel(src,cv.CV_64F,1,0,ksize=5)#y边缘
# dst1 = cv.Sobel(src,cv.CV_64F,0,1,ksize=5)#x边缘
# dst = dst+dst1 #dst = cv.add(dst,dst1)
# #Scharr(3*3)<-Sobel的ksize=-1的特殊
# dst = cv.Scharr(src,cv.CV_64F,1,0)#y边缘
# dst1 = cv.Scharr(src,cv.CV_64F,0,1)#x边缘
# dst = dst+dst1 #dst = cv.add(dst,dst1)
# #Laplacian(同时x,y检测,缺降噪)
# dst = cv.Laplacian(src,cv.CV_64F,ksize=5)
# #Canny
# dst = cv.Canny(src,100,150)

# cv.imshow('src',src)
# cv.imshow('dst',dst)
# cv.waitKey(0)
# cv.destroyAllWindows()



#8,
# from concurrent.futures import thread
# import cv2 as cv
# import numpy as np
# from sklearn.preprocessing import KernelCenterer
# src = cv.imread('309_boys.jpg')
# #GRAY
# src = cv.cvtColor(src,cv.COLOR_BGR2GRAY)
# #常规全局阈值BINARY
# ret,dst = cv.threshold(src,100,255,cv.THRESH_BINARY)
# #自适应阈值
# dst = cv.adaptiveThreshold(src,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,0)
# #腐蚀
# Kernel = np.ones((3,3),np.uint8)#可用cv.getStructuringElement(cv.MORPH_RECT,(3,3))
# dst = cv.erode(src,Kernel,iterations=2)
# #膨胀
# Kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))#Kernel = np.zeros((3,3),np.uint8)#锚点一直为0,无法作用
# dst = cv.dilate(src,Kernel,iterations=2)
# #开运算
# Kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
# dst = cv.morphologyEx(src,cv.MORPH_OPEN,Kernel)
# #闭运算
# Kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
# dst = cv.morphologyEx(src,cv.MORPH_CLOSE,Kernel)
# #梯度
# Kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
# dst = cv.morphologyEx(src,cv.MORPH_GRADIENT,Kernel)
# #顶帽
# Kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
# dst = cv.morphologyEx(src,cv.MORPH_TOPHAT,Kernel)
# #黑帽
# Kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
# dst = cv.morphologyEx(src,cv.MORPH_BLACKHAT,Kernel)


# cv.imshow('src',src)
# cv.imshow('dst',dst)
# cv.waitKey(0)
# cv.destroyAllWindows()



#9,
# import cv2 as cv
# from matplotlib.pyplot import contour
# import numpy as np

# def drawshape(src1,points1):
#     i = 0
#     while i<len(points1):
#         if (i == len(points1)-1):
#             x,y = points1[i][0]
#             x1,y1 = points1[0][0]
#             cv.line(src1,(x,y),(x1,y1),(255,0,0),4)
#         else:    
#             x,y = points1[i][0]
#             x1,y1 = points1[i+1][0]
#             cv.line(src1,(x,y),(x1,y1),(255,0,0),4)
#         i = i+1
# src = cv.imread('309_boys.jpg')
# #灰度和二值化
# src1 = cv.cvtColor(src,cv.COLOR_BGR2GRAY)
# ret,dst = cv.threshold(src1,150,255,cv.THRESH_BINARY)
# #轮廓查找和绘制
# contours,hierarchy = cv.findContours(dst,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)#无法画出
# #cv.drawContours(src,contours,-1,(0,0,255))//-1指定全部轮廓 index 
# #面积和周长
# # area = cv.contourArea(contours[10])
# # len = cv.arcLength(contours[10],True)
# # print("area=%d,len=%d"%(area,len))

# #多边形逼近
# # e = 10
# # approx = cv.approxPolyDP(contours[10],e,True)#轮廓参考点
# # drawshape(src,approx)
# #凸包
# # hull = cv.convexHull(contours[10])#轮廓参考点
# # drawshape(src,hull)
# # cv.line(src,(20,100),(20,400),(255,0,0),4)

# #最小外接矩阵
# # r = cv.minAreaRect(contours[1])#angle,x,y,w,h
# # box = cv.boxPoints(r)#x,y,w,h#取其中部分data
# # box = np.int0(box)//转换
# # cv.drawContours(src,[box],0,(0,255,0),2)
# #最大外接矩阵
# # x,y,w,h = cv.boundingRect(contours[1])#x,y,w,h
# # cv.rectangle(src,(x,y),(x+w,y+h),(255,0,0),2)

# cv.imshow('src',src)
# cv.imshow('dst',dst)
# cv.waitKey(0)
# cv.destroyAllWindows()

#10,
# from pickle import TRUE
# from xml.sax.handler import all_properties
# import cv2 as cv
# from cv2 import drawKeypoints
# from cv2 import Algorithm
# import numpy as np
# from sqlalchemy import true


# blocksize = 2
# ksize = 3
# k = 0.03
# src = cv.imread('cool_girl.jpg')
# src1 = cv.imread('cool_girl_face.jpg')
# dst=cv.cvtColor(src,cv.COLOR_BGR2GRAY)
# dst1=cv.cvtColor(src1,cv.COLOR_BGR2GRAY)
# #Harris角点
# # f1ang = cv.cornerHarris(dst,blocksize,ksize,k)#权衡系数
# # src[f1ang>0.01*f1ang.max()] = [0,0,255]#对角点取阈值
# #Shi-Tomasi角点
# # corners = cv.goodFeaturesToTrack(dst,100,0.01,20)
# # corners = np.int0(corners)
# # for i in corners:
# #     x,y = i.ravel()#多维转一维 
# #     cv.circle(src,(x,y),3,(0,255,0),-1)
# #SIFT角点
# # sift = cv.xfeatures2d.SIFT_create()#SIFT对象
# # #kp = sift.detect(dst,None,)#关键点
# # kp , des = sift.detectAndCompute(dst,None)#关键点和描述子
# # cv.drawKeypoints(dst,kp,src)#绘制
# #SURF角点
# # surf = cv.xfeatures2d.SURF_create()#SURF对象
# # kp , des = surf.detectAndCompute(dst,None)#关键点和描述子
# # cv.drawKeypoints(dst,kp,src)#绘制dst的kp->src 
# #ORB特征点
# # orb = cv.ORB_create()
# # kp,des = orb.detectAndCompute(dst,None)
# # cv.drawKeypoints(dst,kp,src)

# #BF特征匹配
# # sift = cv.xfeatures2d.SIFT_create()
# # kp , des = sift.detectAndCompute(dst,None)#获取特征点和描述子
# # kp1 , des1 = sift.detectAndCompute(dst1,None)#获取特征点和描述子
# # bf = cv.BFMatcher(cv.NORM_L1)#匹配器
# # match = bf.match(des,des1)#描述子匹配
# # dst3 = cv.drawMatches(src,kp,src1,kp1,match,None)

# #FLANN特征匹配
# good = []
# sift = cv.xfeatures2d.SIFT_create()
# kp , des = sift.detectAndCompute(dst,None)#获取特征点和描述子
# kp1 , des1 = sift.detectAndCompute(dst1,None)#获取特征点和描述子

# index_params = dict(algorithm=1,tree=5)#字典参数1(算法,层数)
# search_parames = dict(checks=50)#字典参数2(遍历次数)
# flann = cv.FlannBasedMatcher(index_params,search_parames)#创建匹配器
# dmatch = flann.knnMatch(des,des1,k=2)#描述子匹配计算
# #i记下distance属性,去特征匹配点
# for i,(m,n) in enumerate(dmatch):#DMach对象:匹配点选取优化 距离,两组匹配点比较选取
#     if m.distance<0.7*n.distance:
#         good.append(m)#记下匹配点索引
# # dst2 = cv.drawMatchesKnn(src,kp,src1,kp1,[good],None)        

# #单应性矩阵
# if len(good)>=4:
#     srcPts =np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1,1,2)#搜索图 
#     dstPts =np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2)#训练图
#     H,_ = cv.findHomography(srcPts,dstPts,cv.RANSAC,5.0)
#     #透视变换
#     h,w = src1.shape[:2]
#     pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)#原图四点位置
#     dst3 = cv.perspectiveTransform(pts,H)#H透视后四点位置
#     cv.polylines(src1,[np.int32(dst3)],True,(0,0,255),4)#多边形绘制
#     print('.....')
# else:
#     print('good is less than 4')
#     exit()
# dst2 = cv.drawMatchesKnn(src,kp,src1,kp1,[good],None)     

# cv.imshow('dst2',dst2)
# cv.waitKey(0)
# cv.destroyAllWindows()


#11,
from pickletools import uint8
import numpy as np
import cv2 as cv
from sympy import print_glsl
from tables import Unknown

#获取背景 
#1,二值化
#2,形态学背景
# src1 = cv.imread('cool_girl.jpg')
# src2 = cv.imread('cool_girl_face.jpg')
# dst1=cv.cvtColor(src1,cv.COLOR_BGR2GRAY)
# dst2=cv.cvtColor(src2,cv.COLOR_BGR2GRAY)
# #提取背景
# ret,thesh = cv.threshold(dst1,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)#二值化(自适应阈值)
# kernel = np.ones((3,3),np.int8)
# open1 = cv.morphologyEx(thesh,cv.MORPH_OPEN,kernel,iterations=2)#开运算
# bg = cv.dilate(open1,kernel,iterations=1)#膨胀
# #提取前景
# dist = cv.distanceTransform(open1,cv.DIST_L2,5)#L1(3),L2(5),距离测算:非0到0梯度
# ret,fg = cv.threshold(dist,0.7*dist.max(),255,cv.THRESH_BINARY_INV)#定下前景
# #未知域
# fg = np.uint8(fg)
# Unknow = cv.subtract(bg,fg)
# #创建连通域
# ret,masker = cv.connectedComponents(fg)#masker为连通的所有区域(fg,Unknow)
# masker = masker+1
# masker[Unknow==255]=0#标记连通域变黑(以这点为分水邻标记点)
# #masker标记点进行分水岭分割
# result = cv.watershed(src1,masker)
# src1[result == -1] = [0,0,255]#fg标红
# #---------------
# class App:
#     startX,startY=0,0#成员变量
#     flag_rect = False
#     rect=(0,0,0,0)
#     def onmouse(self,event,x,y,flags,param):#self=this指针
#         if event == cv.EVENT_LBUTTONDOWN:
#             self.startX = x
#             self.startY = y
#             self.flag_rect=True
#             #print('EVENT_LBUTTONDOWN')
#         elif event == cv.EVENT_LBUTTONUP:
#             self.flag_rect=False
#             cv.rectangle(self.img,(self.startX,self.startY),(x,y),(0,0,255),3)
#             #获取rect
#             self.rect = (min(self.startX,x),min(self.startY,y),abs(self.startX-x),abs(self.startY-y))
#            # print('EVENT_LBUTTONUP')
#         elif event == cv.EVENT_MOUSEMOVE:
#             if self.flag_rect==True:
#                 self.img = self.img2.copy()#2原始备份->复制->绘制
#                 cv.rectangle(self.img,(self.startX,self.startY),(x,y),(0,255,0),2)
#             #print('EVENT_MOUSEMOVE')    
#     def run(self):
#         #print("run")

#         cv.namedWindow('input')
#         cv.setMouseCallback('input',self.onmouse)#绑定窗口事件

#         self.img = cv.imread('309_boys.jpg')#隐式创建成员变量
#         self.img2 = self.img.copy()#1原始备份
#         #获取mask:0-保持掩码区图像
#         self.mask = np.zeros(self.img.shape[:2],dtype=np.uint8)#单通道
#         self.output = np.zeros(self.img.shape,np.uint8)

#         while(1):
#             cv.imshow('input',self.img)
#             cv.imshow('output',self.output)     
#             k = cv.waitKey(100)
#             if k==27:
#                 break
#             #分离
#             if k == ord('g'):
#                 bgdmodel = np.zeros((1,65),np.float64)
#                 fgdmodel = np.zeros((1,65),np.float64)
#                 cv.grabCut(self.img2,self.mask,self.rect,bgdmodel,fgdmodel,1,cv.GC_INIT_WITH_RECT)
#             #前景1/3
#             mask2 = np.where((self.mask==1)|(self.mask==3),255,0).astype(np.uint8)
#             #输出
#             self.output = cv.bitwise_and(self.img2,self.img2,mask=mask2)

# App().run()#类的实例化---grabCut ,对象初始化                           
# #----------
# mean_img = cv.pyrMeanShiftFiltering(src1,20,30)#强化边缘
# imgcanny = cv.Canny(mean_img,150,300)#找出边缘

# contours, _ = cv.findContours(imgcanny,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)#找轮廓
# cv.drawContours(src1,contours,-1,(0,255,0),2)#画出轮廓
# # cv.imshow('bg',bg)
# # cv.imshow('fg',fg)
# # cv.imshow('Unknow',Unknow)
# cv.imshow('src1',src1)
# cv.imshow('imgcanny',imgcanny)
# cv.waitKey(0)
# cv.destroyAllWindows()


#12,
# import cv2 as cv
# import numpy as np

# # src1 = cv.imread('cool_girl.jpg')
# # mask = cv.imread('cool_girl_face.jpg',0)
# # dst = cv.inpaint(src1,mask,2,cv.INPAINT_TELEA)#图像修复,大小要一样

# cap = cv.VideoCapture('out1.mp4')
# #创建对象
# #mog = cv.bgsegm.createBackgroundSubtractorMOG()
# mog = cv.createBackgroundSubtractorMOG2()
# #mog = cv.bgsegm.createBackgroundSubtractorGMG(1)
# while True:
#     ret,frame = cap.read()
#     fgmask = mog.apply(frame)

#     cv.imshow('fgmask',fgmask)
#     # cv.imshow('dst',dst)
#     k = cv.waitKey(10)
#     if k==27:
#         break
# cap.relase()
# cv.destroyAllWindows()


#13,
# import cv2 as cv
# import numpy as np
# # import pytessaeract   #tessaeract库

# #创建Haar级联器
# facer = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
# eyer = cv.CascadeClassifier('haarcascade_eye.xml')
# #导入人脸识别图片并将其灰度化
# img = cv.imread('cool_girl.jpg')
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# #进行人脸识别,[返回[x,y,w,h]]
# faces = facer.detectMultiScale(gray,1.1,5)
# i=0
# for (x,y,w,h) in faces:
#     cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
#     roi_img = img[y:y+h,x:x+w]#取出人脸(行y,列x)
#     eyes = eyer.detectMultiScale(roi_img,1.1,5)
#     for (x,y,w,h) in eyes:
#         cv.rectangle(roi_img,(x,y),(x+w,y+h),(0,255,0),2)
#     i = i+1
#     winname = 'face'+str(i)    
#     cv.imshow(winname,roi_img)   

# cv.imshow('img',img)
# cv.waitKey(0)
# cv.destroyAllWindows()

#14,
# import cv2 as cv
# import numpy as np
# from cv2 import dnn

# 1,导入模型，创建神经网络
# config =
# model =
# net = dnn.readNetFromCaffe(config,model)
# 2,读图片，转成张量
# img = cv.imread()
# blob = dnn.blobFromImage(img,1.0,(224,224),(104,117,123))
# 3,张量送入网络，并执行预测
# net.setInput(blob)
# r = net.forward()#预测结果描述
# 4,将预测结果匹配类别目录
# classes = []
# patf = 
# with open(path,'rt') as f:#读txt文件
#   classes = [x[x.find("")+1:] for x in f]
##5,匹配
# z = list(range(3))
##r结果排序(匹配程度大到小)!即匹配的类别进行排序--类别和描述,
# order = sorted(r[0],reverse = True) 
# for i in list(range(0,3)):             #遍历前三项进行匹配
#   z[i] = np.where(r[0]==order[i][0][0])#取前三项匹配
#   print('第',i+1'项匹配:',classes[z[i]],end='')#ps:只是把匹配程度可以打印出来
#   print('类所在行',z[i]+1,',可能性:'，order[i])












