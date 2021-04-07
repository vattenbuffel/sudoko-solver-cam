import cv2
import numpy as np
from line_test import extend_lines, construct_line, intersection_lines
from solver import solve
from network import load_model
from corner_detection_test2 import extract_board

def create_base_board():
    # Create a base sudoko board
    width = width_max
    height = height_max
    base_board = np.zeros((height, width, 3), dtype='uint8')
    base_board[:,:] = np.array([255,255,255])
    cell_height = height//9
    cell_width = width//9

    # Add boarders
    board_thickness = 1
    black = np.array([0,0,0], dtype='uint8')
    for row in range(9):
        base_board[row*cell_height:row*cell_height+board_thickness, :] = black

    for row in range(1, 9+1):
        base_board[row*cell_height-board_thickness:row*cell_height, :] = black

    # Add col walls
    for col in range(9):
        base_board[:, col*cell_width:col*cell_width+board_thickness] = black

    for col in range(1, 9+1):
        base_board[:, col*cell_width-board_thickness:col*cell_width] = black

    return base_board

def combine_images(imgs, individual_max_width=1280, individual_max_height=720, max_width=1500, max_height=700):
    assert len(imgs) > 0, f"Can't display {len(imgs)} amount of images. Must be at least 1."
    
    # Reshape all images to be the correct size
    reshape_imgs = []
    for img in imgs:
        # Make grey_scale_imgs have 3 channels
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        reshape_imgs.append(cv2.resize(img, (individual_max_width, individual_max_height)))
        
    combined_img = np.hstack(reshape_imgs)

    # Reshape combined_img to be of correct result
    if combined_img.shape[0] > max_height or combined_img.shape[1] > max_width:
        desired_shape = [max_width, max_height]
        if combined_img.shape[0] > max_height:
            desired_shape[1] = max_height
        if combined_img.shape[1] > max_width:
            desired_shape[0] = max_width
        
        combined_img = cv2.resize(combined_img, tuple(desired_shape))
        

    return combined_img

def get_edges(img):
    # Canny filter
    img_edges = cv2.Canny(img, 50, 200, None, 3) #cv2.Canny(img, 60, 120)
    return img_edges

def get_lines(img):
    linesP = cv2.HoughLinesP(img, 1, np.pi / 180, 50, None, 50, 10)
    if linesP is None:
        return None
    linesP = linesP.reshape(-1,4)
    return linesP
        
def draw_lines(img, lines):
    if not len(img.shape) == 3:
        img_lines = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_lines = np.copy(img)
    for i in range(0, len(lines)):
        l = lines[i]
        cv2.line(img_lines, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
    return img_lines

def warp_lines(lines, h):
    warped_lines = []
    for line in lines:
        p0 = np.array((list(line[:2]) + [1]))
        p1 = np.array((list(line[2:4]) + [1]))

        p0_warped = h@p0
        p1_warped = h@p1
        
        p0_warped[:2]/=p0_warped[2]
        p1_warped[:2]/=p1_warped[2]

        warped_lines.append(np.array([p0_warped[0], p0_warped[1], p1_warped[0], p1_warped[1]], dtype='int'))

    return warped_lines

def calc_km(p0, p1):
        k = (p1[1]-p0[1]) / (p1[0]-p0[0])
        m = p1[1] - k*p1[0]
        return k, m

def get_kp(img):
    # mouse callback function
    def draw_circle(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(img,(x,y),3,(255,0,0),-1)
            kps.append(np.array([x,y]))
    
    kps = []
    cv2.namedWindow('keypoints')
    cv2.setMouseCallback('keypoints', draw_circle)
    while(1):
        cv2.imshow('keypoints', img)
        if cv2.waitKey(20) & 0xFF == 27:
            break
        if len(kps) == 4:
            break
    cv2.destroyAllWindows()
    return kps

def calc_angle_of_lines(lines):
    angle_of_lines = {}
    for line in lines:
        assert line[2]-line[0] >= 0 , f"P1 must be to the right of P0"
        angle = np.arctan2(line[3]-line[1], line[2]-line[0])
        angle_of_lines[tuple(line)] = angle
    return angle_of_lines
    
def eliminate_unrelated_lines(lines, epsilon = 10*np.pi/180, n_neighbours=20):
    angle_of_lines = calc_angle_of_lines(lines)
    good_lines = []
    
    for line in angle_of_lines:
        angle = angle_of_lines[line]
        neighbours = 0
        for angle_ in angle_of_lines.values():
            if np.abs(angle-angle_) < epsilon:
                neighbours += 1

        if neighbours >= n_neighbours:
            good_lines.append(line)
    
    return good_lines

def eliminate_duplicate_lines(lines, epsilon_angle =  10*np.pi/180, intersection_distance=500):
    angle_of_lines = calc_angle_of_lines(lines)
    good_lines = {}
    
    for line in angle_of_lines:
        angle = angle_of_lines[line]
        duplicate = False

        for line_ in good_lines:
            angle_ = angle_of_lines[line_]
            if np.abs(angle-angle_) < epsilon_angle:
                x,y = intersection_lines(line, line_)
                if x is not None and (x**2+y**2)**0.5 < intersection_distance:
                    duplicate = True
                    break

        if not duplicate:
            good_lines[line] = angle

    return list(good_lines.keys())

def eliminate_duplicated_line_after_warp(lines, epsilon_angle =  10*np.pi/180, distance_to_remove=10):
    # Figure out which lines are vertical and horizontal
    horizontal_lines = {}
    vertical_lines = {}
    angle_of_lines = calc_angle_of_lines(lines)
    
    for line in angle_of_lines:
        if np.abs(angle_of_lines[line]) < epsilon_angle:
            horizontal_lines[line[1]] = line

        if np.abs(np.pi/2 - np.abs(angle_of_lines[line])) < epsilon_angle:
            vertical_lines[line[0]] = line
    
    # If there aren't enough lines:
    if len(list(horizontal_lines.keys())) < 10 or len(list(vertical_lines.keys())) < 10:
        return None
    # Remove duplicated vertical lines
    intersections = {}
    h_line0 = horizontal_lines[list(horizontal_lines.keys())[0]]
    h_line1 = horizontal_lines[list(horizontal_lines.keys())[-1]]
    vert_line_to_remove = set() # TODO: Maybe check which lines are the straightest and keep that
    for key in vertical_lines:
        v_line = vertical_lines[key]
        intersections[key] = intersection_lines(h_line0, v_line), intersection_lines(h_line1, v_line)
        
    for key in intersections:
        if key in vert_line_to_remove:
            continue
        int11 = np.array(intersections[key][0])
        int12 = np.array(intersections[key][1])
        for key_ in intersections:
            if key == key_:
                continue
            int21 = np.array(intersections[key_][0])
            int22 = np.array(intersections[key_][1])
            distance = []
            distance.append(np.linalg.norm(int11-int21))
            distance.append(np.linalg.norm(int11-int22))
            distance.append(np.linalg.norm(int12-int21))
            distance.append(np.linalg.norm(int12-int22))
            if np.any(np.array(distance) < distance_to_remove):
                vert_line_to_remove.add(key_)
    
    for key in vert_line_to_remove:
        del vertical_lines[key]

    intersections = {}
    v_line = vertical_lines[list(vertical_lines.keys())[0]]
    hort_line_to_remove = set()
    for key in horizontal_lines:
        h_line = horizontal_lines[key]
        intersection = intersection_lines(h_line, v_line)
        assert intersection[0] is not None
        intersections[key] = intersection
        
    for key in intersections:
        if key in hort_line_to_remove:
            continue
        int1 = np.array(intersections[key])
        for key_ in intersections:
            if key == key_:
                continue
            int2 = np.array(intersections[key_])
            distance = np.linalg.norm(int1-int2)
            if distance < distance_to_remove:
                hort_line_to_remove.add(key_)
        
    for key in hort_line_to_remove:
        del horizontal_lines[key]

    return list(horizontal_lines.values()), list(vertical_lines.values())

def show_lines(text, lines, img):
    assert len(lines[0]) == 4, "The shape of lines must be (-1,4)."
    img_with_line_to_remove = draw_lines(np.copy(img), [np.array(line, dtype='int') for line in lines])
    cv2.imshow(text, img_with_line_to_remove)
    cv2.waitKey(0)
    cv2.destroyWindow(text)
    test = 5

def lines_forming_sudoko(lines, height_max, width_max, epsilon_angle = 10*np.pi/180, img=None, max_increase_factor=1.0):
    horizontal_lines = {}
    vertical_lines = {}

    angle_of_lines = calc_angle_of_lines(lines)
    
    for line in angle_of_lines:
        if np.abs(angle_of_lines[line]) < epsilon_angle:
            horizontal_lines[line[0]] = line

        if np.abs(np.pi/2 - np.abs(angle_of_lines[line])) < epsilon_angle:
            vertical_lines[line[1]] = line

    # Remove all lines which intersect with the board outside of the image
    v0 = [0,0,0,height_max*max_increase_factor]
    h0 = [0,0,width_max*max_increase_factor,0]
    hor_line_to_remove = []
    for key in horizontal_lines:
        x,y = intersection_lines(horizontal_lines[key], v0)
        if x < 0 or x >= width_max or y < 0 or y >= height_max:
            hor_line_to_remove.append(key)
            
            # Show the line about to be removed
            if img is not None:
                show_lines("horizontal line to remove", [horizontal_lines[key]], img)

    for key in hor_line_to_remove: 
        del horizontal_lines[key]

    vert_line_to_remove = []
    for key in vertical_lines:
        x,y = intersection_lines(vertical_lines[key], h0)
        if x < 0 or x >= width_max or y < 0 or y >= height_max:
            vert_line_to_remove.append(key)
            
            # Show the line about to be removed
            if img is not None:
                show_lines("vertical line to remove", [vertical_lines[key]], img)

    for key in vert_line_to_remove: 
        del vertical_lines[key]

    return list(horizontal_lines.values()), list(vertical_lines.values())
    
def extract_cells(v_lines, h_lines):
    h_lines_dict = {line[1]:line for line in h_lines}
    v_lines_dict = {line[0]:line for line in v_lines}

    sorted_h_lines = np.sort(list(h_lines_dict.keys()))
    sorted_v_lines = np.sort(list(v_lines_dict.keys()))

    sorted_h_lines = [h_lines_dict[key] for key in sorted_h_lines]
    sorted_v_lines = [v_lines_dict[key] for key in sorted_v_lines]

    cells = {}
    for i in range(len(sorted_h_lines)-1):
        h_line_first = sorted_h_lines[i]
        h_line_second = sorted_h_lines[i+1]
        for j in range(len(sorted_v_lines)-1):
            v_line_first = sorted_v_lines[j]
            v_line_second = sorted_v_lines[j+1]

            int1 = intersection_lines(h_line_first, v_line_first)
            assert int1[0] is not None
            int2 = intersection_lines(h_line_second, v_line_second)
            assert int2[0] is not None

            cells[(i,j)] = [int1[0], int1[1], int2[0], int2[1]]

    return cells

def show_all_cells(cells, img, waitKey=0):
    cv2.namedWindow("cells")
    for p in cells.values():
        p0 = np.array([p[0], p[1]], dtype='int')
        p1 = np.array([p[2], p[3]], dtype='int')
        cell = img[p0[1]:p1[1], p0[0]:p1[0]]
        cv2.imshow("cells", cell)
        key = cv2.waitKey(waitKey)
        if key == 115: # s
            name = "./img/" + str(np.random.random()) + ".png"
            cv2.imwrite(name, cell)
        if key == 113: #s
            return

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
            name = "./img/numbers/" + str(val) + "/" + str(val) + "-" + str(max_vals[val]) + ".png"
            cv2.imwrite(name, cell)
        
        if key == 98: #b:
            max_vals[0] += 1
            name = "./img/numbers/" + "blank" + "/" + "blank" + "-" + str(max_vals[0]) + ".png"
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
        names = os.listdir("./img/numbers/" + str(i))
        max_n[i] = max_val_in_file_name(names, str(i)+'-')

    max_n[0] = max_val_in_file_name(os.listdir("./img/numbers/blank"), "blank-") # 0=blank
    
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

def build_board(cells, img, digit_recognizer, show=True, crop_val_factor=0.15, blank_threshold=0.99):
    board = np.zeros((9,9), dtype='int')
    for cell in cells:
        p = cells[cell]
        p0 = np.array([p[0], p[1]], dtype='int')
        p1 = np.array([p[2], p[3]], dtype='int')
        img_digit = img[p0[1]:p1[1], p0[0]:p1[0]]
        y_crop, x_crop = int(img_digit.shape[0]*crop_val_factor), int(img_digit.shape[1]*crop_val_factor)
        img_digit = img_digit[y_crop:-y_crop,x_crop:-x_crop]

        # Check if empty cell
        if np.prod(img_digit.shape) == 0:
            return None

        # Change black to black and white to white
        img_digit = cv2.adaptiveThreshold(cv2.cvtColor(img_digit, cv2.COLOR_BGR2GRAY),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,3)
        img_digit[img_digit==255] = 1
        img_digit[img_digit==0] = 255
        img_digit[img_digit==1] = 0

        # Check if the center 50 % of the img is almost empty
        height, width = img_digit.shape[:2]
        center_img = img_digit[int(height*0.25):-int(height*0.25),int(width*0.25):-int(width*0.25)]
        if np.sum(center_img/255)/np.prod(center_img.shape) > blank_threshold:
            val = 0
        else:
            val = digit_recognizer.predict_on_image(img_digit)

        if show:
            print(f"Predicted {val}")
            cv2.imshow("prediction", img_digit)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        board[cell] = val
    
    return board

def extract_numbers_and_cells(img):
    img_sudoko_color = img
    img_sudoko_color, M = extract_board(img)    
    img_sudoko = cv2.cvtColor(img_sudoko_color, cv2.COLOR_BGR2GRAY)

    img_edges = get_edges(img_sudoko)

    lines = get_lines(img_edges)
    if lines is None or len(lines) < 50:
        return None
    img_lines = draw_lines(np.copy(img_edges), lines)

    # img = combine_images([img_sudoko_color, img_edges, img_lines])
    # cv2.imshow("board", img)
    # cv2.waitKey(0)

    # Clean some lines
    lines = [construct_line(line[:2], line[2:]) for line in lines]
    lines = extend_lines(lines, x_max=img_sudoko_color.shape[1], y_max=img_sudoko_color.shape[0])
    lines = np.ceil(lines).astype('int')
    assert np.all(np.diff(np.array(lines)[:,(0,2)],axis=1) >= 0), f"P1 should be to the right of P0."
    # cv2.imshow("extended lines", draw_lines(np.copy(img_edges), lines))
    # cv2.waitKey(0)
    lines = eliminate_unrelated_lines(lines)
    # cv2.imshow("removed unrelated lines", draw_lines(np.copy(img_edges), lines))
    # cv2.waitKey(0)
    lines = eliminate_duplicate_lines(lines)
    # cv2.imshow("removed duplicated lines before warp", draw_lines(np.copy(img_edges), lines))
    # cv2.waitKey(0)
    if len(lines) == 0:
        return None

    res = eliminate_duplicated_line_after_warp(lines)
    if res is None:
        return None
    h_lines, v_lines = res

    # img_sudoko_lines = draw_lines(img_sudoko, h_lines)
    # img_sudoko_lines = draw_lines(img_sudoko_lines, v_lines)
    # cv2.imshow("removed duplicated lines after warp", img_sudoko_lines)
    # cv2.waitKey(0)

    # img = combine_images([img_lines, img_sudoko_lines])
    # cv2.imshow("board", img)
    # cv2.waitKey(0)

    cells = extract_cells(v_lines, h_lines)
    good_cells = set()
    for i in range(9):
        for j in range(9):
            good_cells.add((i,j))
    for cell in cells:
        if not cell in good_cells:
            return None
    # cv2.destroyAllWindows()
    return cells, img_sudoko_color

if __name__ == '__main__':
    # img_name="./img/sudoko1.png"
    # img = cv2.imread(img_name)
    cap = cv2.VideoCapture(0)
    digit_recognizer = load_model()
    while True:
        ret, img = cap.read()
        res = extract_numbers_and_cells(img)
        if res is not None:
            cells, img_warped = res
            board = build_board(cells, img_warped, digit_recognizer, show=False)
            try:
                solve(board)
            except:
                print("Invalid board")
            combined = combine_images([img, img_warped])
        else:
            # combined = combine_images([img, np.zeros((480,640,3))])
            combined = combine_images([img, img])
        cv2.imshow("input, extracted", combined)
        cv2.waitKey(1)

    
    # 


    # generate_training_data()

