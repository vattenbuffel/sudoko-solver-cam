import cv2 
import numpy as np
import os

# img_size = 50
img_size = -1
pixel_threshold_value = 70
max_line_thickness = 3
line_probability = 0.05
# line_probability = -1
p_salt_and_pepper = 0.005


base_path = "./img/datasets/combined/training-set/"
# base_path = "./img/datasets/combined/validation-set/"


def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def pre_prcess_num():
    folders = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "blank"]
    for i in folders:
        num_path = base_path + i + "/"
        for path in os.listdir(num_path):
            img_path = num_path + path
            img = cv2.imread(img_path)

            # Reshape
            if img_size > 0:
                img = cv2.resize(img, (img_size, img_size))
                # cv2.imshow("Resized", img)
                # cv2.waitKey(0)

            # Change all black to black and all white to white
            if pixel_threshold_value > -1:
                img[np.any(img[:,:] < np.array([pixel_threshold_value,pixel_threshold_value,pixel_threshold_value]), axis=2)] = 0
                img[np.any(img[:,:] > np.array([pixel_threshold_value,pixel_threshold_value,pixel_threshold_value]), axis=2)] = 255
                # cv2.imshow("white and black", img)
                # cv2.waitKey(0)
            
            # Add lines
            if np.random.random() < line_probability: #v0
                x0 = np.random.randint(-10,10)
                y0 = np.random.randint(-10,10) 
                x1 = np.random.randint(-10,10) + img_size
                y1 = np.random.randint(-10,10) 
                thickness = np.random.randint(1,max_line_thickness+1)
                img = cv2.line(img, (x0, y0), (x1, y1), (0,0,0), thickness, cv2.LINE_AA)
                # cv2.imshow("v0", img)
                # cv2.waitKey(0)

            if np.random.random() < line_probability: #v1
                x0 = np.random.randint(-10,10) + img_size
                y0 = np.random.randint(-10,10) + img_size
                x1 = np.random.randint(-10,10) + img_size
                y1 = np.random.randint(-10,10) 
                thickness = np.random.randint(1,max_line_thickness+1)
                img = cv2.line(img, (x0, y0), (x1, y1), (0,0,0), thickness, cv2.LINE_AA)
                # cv2.imshow("v1", img)
                # cv2.waitKey(0)

            if np.random.random() < line_probability: #h0
                x0 = np.random.randint(-10,10)
                y0 = np.random.randint(-10,10) + img_size
                x1 = np.random.randint(-10,10) 
                y1 = np.random.randint(-10,10) 
                thickness = np.random.randint(1,max_line_thickness+1)
                img = cv2.line(img, (x0, y0), (x1, y1), (0,0,0), thickness, cv2.LINE_AA)
                # cv2.imshow("h0", img)
                # cv2.waitKey(0)

            if np.random.random() < line_probability: #h1
                x0 = np.random.randint(-10,10) + img_size
                y0 = np.random.randint(-10,10) + img_size
                x1 = np.random.randint(-10,10) + img_size
                y1 = np.random.randint(-10,10) 
                thickness = np.random.randint(1,max_line_thickness+1)
                img = cv2.line(img, (x0, y0), (x1, y1), (0,0,0), thickness, cv2.LINE_AA)
                # cv2.imshow("h1", img)
                # cv2.waitKey(0)

            
            # Add salt and pepper noise
            img = sp_noise(img, p_salt_and_pepper)
            # cv2.imshow("Salt and pepper", img)
            # cv2.waitKey(0)


            # cv2.imshow("test", img)
            # cv2.waitKey(0)
            cv2.imwrite(img_path, img)
        
        print(f"Done with {num_path}")



pre_prcess_num()






