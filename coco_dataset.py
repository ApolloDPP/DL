# -*- coding: utf-8 -*-
"""
Created on Wed Aug  19 14:04:33 2020

@author: ApolloDPP
"""

"""
本实验中使用的coco数据集为train2017以及对应的annotation_train2017
"""
import cv2
import os
import json
import pycocotools.coco as coco


#定义函数将标注信息画在原图上：包含bbox和class
def annotation2img(img_path,annotation_path):
        
        #导入图片与其相对应的标注信息
        img=cv2.imread(img_path)
        img_ann=json.load(open(annotation_path, 'r'))
        
        #获取图片名方便获取img_id与标注信息id的匹配
        img_name=os.path.basename(img_path)
        
        #获取img-id
        img_id=None
        for img in img_ann['images']:
                if img['file_name']==img_name:
                        img_id=img['id']
                        break
                
        #将img_id和标注信息id匹配
        coco_ann = coco(annotation_path)
        annIds = coco_ann.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = coco_ann.loadAnns(annIds)
        
        #作图
        num=len(annIds)
        for i in range(num):
                x, y, w, h = int(anns[i]['bbox'])    #获取标注信息中的bbox信息
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)  #利用bbox对角线信息作图
                categor_id = anns[i]['category_id']
                category = coco_ann.loadCats(categor_id)
                cv2.putText(img, category[0]['name'], (x+10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                return img
        
img_path='D:\cocodata\train2017(1)\train2017\000000000081.jpg'
annotation_path='D:\cocodata\annotations_trainval2017(1)\annotations\instances_train2017.json'              

img = annotation2img(img_path,annotation_path)
cv2.imshow('img', img)
cv2.waitKey(0)               
                
   
        
