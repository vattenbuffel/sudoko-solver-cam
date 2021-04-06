import cv2 
import numpy as np
import os


pixel_threshold_value = 125
base_path = "./img/datasets/combined/training-set/"
# base_path = "./img/datasets/kaggle/"

for i in range(1,10):
    num_path = base_path + str(i) + "/"
    for path in os.listdir(num_path):
        img_path = num_path + path
        img = cv2.imread(img_path)
        img[img < pixel_threshold_value] = 0
        img[img > pixel_threshold_value] = 255
        black = img==0
        white = img==255
        img[white] = 0 
        img[black] = 255
        

        # cv2.imshow("test", img)
        # cv2.waitKey(0)
        cv2.imwrite(img_path, img)
    
    print(f"Done with {num_path}")

num_path = base_path +  "blank/"
for path in os.listdir(num_path):
    img_path = num_path + path
    img = cv2.imread(img_path)
    img[img < pixel_threshold_value] = 0
    img[img > pixel_threshold_value] = 255
    black = img==0
    white = img==255
    img[white] = 0 
    img[black] = 255
    

    # cv2.imshow("test", img)
    # cv2.waitKey(0)
    cv2.imwrite(img_path, img)

print(f"Done with {num_path}")








