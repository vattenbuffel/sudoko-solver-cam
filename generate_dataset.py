import cv2 
import numpy as np
import os

img_size = 50
thickness_min = 1
thickness_max = 2
font = cv2.FONT_HERSHEY_DUPLEX 
font_scale_min = 1 
font_scale_max = 1.5
min_rotation = -30
max_rotation = 30
n_img = 1000

base_path = "./img/datasets/generated/"
for i in range(1,10):
    num_path = base_path + str(i) + "/"
    num = str(i)
    for j in range(n_img):
        img = np.zeros((img_size, img_size, 3), dtype='uint8')
        img[:,:,:] = 1
        font_scale_ = np.random.uniform(font_scale_min, font_scale_max)
        x0,y0 = np.random.randint(-5,30), np.random.randint(-10,5) + img_size
        thickness = np.random.randint(thickness_min, thickness_max+1)
        cv2.putText(img, num, (x0,y0), font, font_scale_, (2, 2, 2), thickness, cv2.LINE_AA)
        
        M = cv2.getRotationMatrix2D((img_size//2, img_size//2), np.random.uniform(min_rotation, max_rotation), 1.0)
        img = cv2.warpAffine(img, M, (img_size, img_size))

        img[img[:,:,:] == 0] = 255
        img[img[:,:,:] == 1] = 255
        img[img[:,:,:] == 2] = 0
        
        cv2.imwrite(num_path + str(j) + ".png", img)
    
    print(f"Done with {i}")







cv2.destroyAllWindows()

