from network import  load_model
import cv2
import os
import numpy as np

digit_recognizer = load_model()


if __name__ == '__main__':
    while True:
        path = "./img/datasets/combined/validation-set/"
        number_to_show = np.random.choice(os.listdir(path))
        path += number_to_show +"/"
        img_path = np.random.choice(os.listdir(path))
        path = path + img_path

        img = cv2.imread(path)
        cv2.imshow("number", img)
        pred = digit_recognizer.predict_on_image(img)
        print(pred)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        test = 5