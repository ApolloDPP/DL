# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:15:00 2020

@author: ApolloDPP
"""



"""
对Anchor的理解
1.Anchor是什么？Anchor的作用？
   Anchor的本质是在原图上一系列大大小小的矩形框，在Faster RCNN中将Anchor在Feature map中
绘制；其作用是为proposal做先验框。
2.Anchor如何参与到运算/训练过程中？
   Anchor是Faster RCNN中首先提出的算法RPN网络中的重要一环，Anchor首先将原图中可能出现的物
体进行覆盖，然后经过RPN网络后筛选出更加检测物体的proposal框。
3.Anchor如何回归？
   feature map是经过4次maxpooling操作得到，与原图相比较尺寸/16，所以映射到原图一个像素增大
16倍。
4.Anchor生成的标准是什么？
   Anchor对于每个像素点都进行生成9个anchors，其尺寸自定义，目的是尽可能覆盖原图中所有检测物体
"""



"""
算法实现
1.在feature map上实现每点对应9个anchors，因为feature map较小，所以将每个anchor尺寸定义较小
A.逐点进行9个anchors的是生成，先定义一个基本的框，然后进行尺寸和比例的变换得到论文中提到的9种
假设feature map尺寸为40*60大小



2.将feature map上anchors映射到原图上，VGG16经过4此maxpooling后尺寸变为/16，所以将映射到原图
时扩展16倍即可。

"""



import numpy as np
import matplotlib.pyplot as plt    #显示图片
import time
import cv2




F_w=60
F_h=40

t=time.time()

def generate_9anchors():
#该函数生成以feature map上（0，0）点为中心的9个anchors,函数输出9个anchors的左上角和右下角坐标值
#基本anchor设置为2*2，ratio=[0.5,1,2],scales=[1,4,16]
          anchor_0=[-1,-1,1,1]  #base anchor
          anchor_1=[-1.414,-0.707,1.411,0.707]  #ratio=0.5
          anchor_2=[-0.707,-1.411,0.707,1.411]  #ratio=2
          anchor_ratio_0=[ anchor_0,anchor_1,anchor_2]  #base_scale
          
          anchor_ratio_1=np.multiply(4,anchor_ratio_0)  #scale=4
          anchor_ratio_2=np.multiply(16,anchor_ratio_0)  #scale=16
          
          anchor_0=[]
          anchor_0=np.vstack((anchor_ratio_0,anchor_ratio_1))
          anchor_0=np.vstack((anchor_0,anchor_ratio_2)) #anchor_0中包含第一个点的9个anchors坐标值
          
          return anchor_0
  
#print(generate_9anchors())
          
def pixel2cord(pixel_cord):
#生成任意像素坐标的9个anchors,坐标值（左上和右下角）
        anchor=generate_9anchors()
        anchor[:,0]+=pixel_cord[0]
        anchor[:,2]+=pixel_cord[0]
        
        anchor[:,1]+=pixel_cord[1]
        anchor[:,3]+=pixel_cord[1]
        
        return anchor
       
        
def cord2wh_leftcord(anchor):
#将据行框的坐标表示转化为plt作图的参数形式：左上角坐标，w,h的值，为9*4大小数组
        wh_leftcord=np.zeros((9,4))  #保存左上角坐标，w,h的值，为9*4大小数组
        wh_leftcord[:,0]=anchor[:,0]
        wh_leftcord[:,1]=anchor[:,1]
        
        wh_leftcord[:,2]=anchor[:,2]-anchor[:,0]
        wh_leftcord[:,3]=anchor[:,3]-anchor[:,1]
        
        return wh_leftcord
        
 
#在feature map上生成anchors     
fig = plt.figure(1)     
plt.axis([-10,70,-10,50])       #plt作图显示坐标范围
ax = fig.add_subplot(111)
ax.set_title('Feature map',fontsize=12,color='b')
rect0= plt.Rectangle((0,0), 60,40, fill=False, edgecolor = 'green',linewidth=5) #绿色框内为feature map框
ax.add_patch(rect0)   
plt.show()

for i in range(2,60,20):        #每隔10个像素作图
        for j in range(2,40,10):
                pixel_cord=[i,j]               
                anchor=pixel2cord(pixel_cord)
                wh_leftcord=cord2wh_leftcord(anchor)
               

                for k in range(0,9):
                        if (wh_leftcord[k,:][2]<F_w and wh_leftcord[k,:][3]<F_h and wh_leftcord[k,:][0]>=0 and wh_leftcord[k,:][1]>=0
                            and wh_leftcord[k,:][0]+wh_leftcord[k,:][2]<F_w and wh_leftcord[k,:][1]+wh_leftcord[k,:][3]<F_h):
                        #此判断语句将超出feature map的anchors舍去
                                rect= plt.Rectangle((wh_leftcord[k,:][0],wh_leftcord[k,:][1]),
                                                    wh_leftcord[k,:][2],wh_leftcord[k,:][3], 
                                                    fill=False, edgecolor = 'red',linewidth=1)
                                ax.add_patch(rect)   
                                plt.show()  
                        else : break
                        


#anchors映射到原图
fig = plt.figure(2)     #在feature map上生成anchor
cat=cv2.imread('cat.jpg')
cat=cv2.resize(cat,(960,640))
cat=cv2.cvtColor(cat,cv2.COLOR_BGR2RGB)

plt.imshow(cat)
plt.show()

plt.axis([-10,1000,-10,700])       #plt作图显示坐标范围
ax = fig.add_subplot(111)
ax.set_title('oroginal',fontsize=12,color='b')
rect0= plt.Rectangle((0,0), 960,640, fill=False, edgecolor = 'green',linewidth=5) #绿色框内为feature map框
ax.add_patch(rect0)   
plt.show()

for i in range(2,60,20):        #每隔10个像素作图
        for j in range(2,40,10):
                pixel_cord=[i,j]               
                anchor=np.multiply(16,pixel2cord(pixel_cord))
                wh_leftcord=cord2wh_leftcord(anchor)
               

                for k in range(0,9):
                        if (wh_leftcord[k,:][2]<960 and wh_leftcord[k,:][3]<640 and wh_leftcord[k,:][0]>=0 and wh_leftcord[k,:][1]>=0
                            and wh_leftcord[k,:][0]+wh_leftcord[k,:][2]<960 and wh_leftcord[k,:][1]+wh_leftcord[k,:][3]<640):
                        #此判断语句将超出feature map的anchors舍去
                                rect= plt.Rectangle((wh_leftcord[k,:][0],wh_leftcord[k,:][1]),
                                                    wh_leftcord[k,:][2],wh_leftcord[k,:][3], 
                                                    fill=False, edgecolor = 'red',linewidth=1)
                                ax.add_patch(rect)   
                                plt.show()  
                        else : break
                        



print(time.time()-t)    #计算一张feature map生成anchors的时间


"""
说明：由于算法的时间复杂度太大，故在feature map测试中每隔20/10个像素点画anchors
"""



        



