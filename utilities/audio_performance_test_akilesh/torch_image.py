import torch
import torchvision
from PIL import Image
import cv2
import os
import timeit
import torchvision.transforms.functional as F
from tqdm import tqdm


tot_time =0 
for i in tqdm(range(100)):
    folder_path = "/media/akilesh/100_sample_image/"
    file_list = os.listdir(folder_path)  
    for fn in file_list:
        file_name = folder_path + fn 
        img = Image.open(file_name)
        if(not img):
            print("Error in image read ",file_name)
            exit(0)
        start = timeit. default_timer()
        # color_jitter = torchvision.transforms.ColorJitter( brightness = 1.4 , contrast = 0.4 , saturation = 1.9, hue = 0.1 )
        # bright = F.adjust_brightness(img, 0.3)
        contrast = F.adjust_contrast(img,0.1)
        # img = color_jitter(img)
        tot_time = tot_time + (timeit. default_timer()-start)
print("Time in microseconds ",(tot_time/100)*1000000)  

