import cv2
import numpy as np


if __name__ == '__main__':

    image = cv2.imread("./img/sudoko5.png")
    transformed = extract_board(image)

    cv2.imshow('transformed', transformed)
    # cv2.imwrite('board.png', transformed)
    cv2.waitKey()