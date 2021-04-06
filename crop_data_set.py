import cv2
import os


base_path = "./img/datasets/combined/validation-set/"
crop_val = 15

for i in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]:
    num_path = base_path + str(i) + "/"
    for path in os.listdir(num_path):
        img_path = num_path + path
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        
        
        crop_img = img[crop_val:-crop_val, crop_val:-crop_val]
        # cv2.imshow("test", crop_img)
        # cv2.waitKey(0)
        cv2.imwrite(img_path, crop_img)














