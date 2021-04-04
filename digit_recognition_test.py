from network import  load_model
import cv2
import os
import numpy as np

digit_recognizer = load_model()

while True:
    path = "./img/numbers/"
    number_to_show = np.random.choice(os.listdir(path))
    path += number_to_show +"/"
    img_path = np.random.choice(os.listdir(path))
    path = path + img_path

    img = cv2.imread(path)
    cv2.imshow("number", img)
    pred = digit_recognizer.predict_on_image(img)
    print(pred)
    cv2.waitKey(0)
    test = 5

