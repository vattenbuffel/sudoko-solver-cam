import os
import cv2
import numpy as np
from main import extract_numbers_and_cells

def show_all_cells_and_store(cells, img, max_vals):
    cv2.namedWindow("cells")
    for p in cells.values():
        p0 = np.array([p[0], p[1]], dtype='int')
        p1 = np.array([p[2], p[3]], dtype='int')
        cell = img[p0[1]:p1[1], p0[0]:p1[0]]
        cv2.imshow("cells", cell)
        key = cv2.waitKey(0)
        if key > 48 and key < 58:
            val = key-48
            max_vals[val] += 1
            name = "./img/datasets/sodoku-numbers/" + str(val) + "/" + str(val) + "-" + str(max_vals[val]) + ".png"
            cv2.imwrite(name, cell)
        
        if key == 98: #b:
            max_vals[0] += 1
            name = "./img/datasets/sodoku-numbers/" + "all-blanks" + "/" + "blank" + "-" + str(max_vals[0]) + ".png"
            cv2.imwrite(name, cell)

        if key == 113: #q
            exit()

def max_val_in_file_name(names, prefix):
    max_val = 0
    len_prefix = len(prefix)
    for name in names:
        n = int(name[len_prefix:-4])
        if n > max_val: max_val = n
    return max_val


def generate_training_data():
    import os
    import traceback
    max_n = {}
    for i in range(1,10):
        names = os.listdir("./img/datasets/sodoku-numbers/" + str(i))
        max_n[i] = max_val_in_file_name(names, str(i)+'-')

    max_n[0] = max_val_in_file_name(os.listdir("./img/datasets/sodoku-numbers/all-blanks/"), "blank-") # 0=blank
    
    sudokos_to_do = os.listdir("./img/sudoko/todo")
    for sudokok in sudokos_to_do:
        name = "./img/sudoko/todo/" +  sudokok
        img = cv2.imread(name)
        cells, img = extract_numbers_and_cells(img)
        try:
            show_all_cells_and_store(cells, img, max_n)
        except:
            traceback.print_exc()
            print("Something went wrong with this image.\nMoving on to the next.")

        os.replace(name, "./img/sudoko/done/" +  sudokok)