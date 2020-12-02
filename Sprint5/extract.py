import cv2 as cv  
import random  
import glob  
import os
from PIL import Image
import shutil

def get_samples(foldername,savePath):
    print('savePath:',savePath)
    if os.path.exists(savePath) is False:
        os.makedirs(savePath)
   
    filenames = os.listdir(foldername)
            
    for filename in filenames:  
        full_path = os.path.join(foldername, filename)
        new_name = filename[:-5]+'.png'
        label_png = os.listdir(full_path)[2]
        #os.rename(os.path.join(filename, label_png),os.path.join(filename, name))
        shutil.copy(os.path.join(full_path, label_png), os.path.join(savePath, label_png))
        os.rename(os.path.join(savePath, label_png),os.path.join(savePath, new_name))
        #print(os.listdir(filename))
        
savePath = './greatwallannotation'  
get_samples('./greatwall',savePath )   
