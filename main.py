from queue import Empty, Full
import queue
import cv2
import numpy as np
from line_math import extend_lines, construct_line, intersection_lines
from solver import solve
from network import load_model
import multiprocessing

def create_board(width, height, board, show=False):
    # Create a base sudoko board
    base_board = np.zeros((height, width, 3), dtype='uint8')
    base_board[:,:] = np.array([255,255,255])
    cell_height = height//9
    cell_width = width//9

    # Add boarders
    board_thickness = 1
    thick_board_thickness = 3
    black = np.array([0,0,0], dtype='uint8')
    for row in range(9):
        thickness = board_thickness
        if row %3 == 0:
            thickness = thick_board_thickness
        if row == 0:
            thickness *= 2
        base_board[row*cell_height:row*cell_height+thickness, :] = black

    for row in range(1, 9+1):
        thickness = board_thickness
        if row % 3 == 0:
            thickness = thick_board_thickness
        if row == 9:
            thickness *= 2
        base_board[row*cell_height-thickness:row*cell_height, :] = black

    # Add col walls
    for col in range(9):
        thickness = board_thickness
        if col %3 == 0:
            thickness = thick_board_thickness
        if col == 0:
            thickness *= 2
        base_board[:, col*cell_width:col*cell_width+thickness] = black

    for col in range(1, 9+1):
        thickness = board_thickness
        if col %3 == 0:
            thickness = thick_board_thickness
        if col == 9:
            thickness *= 2
        base_board[:, col*cell_width-thickness:col*cell_width] = black

    # Add numbers
    font = cv2.FONT_HERSHEY_DUPLEX 
    font_scale = 1
    thickness = 1
    x_offset = 25
    y_offset = cell_height - 15
    for y, row in enumerate(board):
        for x, cell in enumerate(row):
            num = cell.num
            x0 = x*cell_width + x_offset
            y0 = y*cell_height + y_offset

            cv2.putText(base_board, str(num), (x0,y0), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
            
            if show:
                cv2.imshow("number", base_board)
                cv2.waitKey(0)


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

def calc_angle_of_lines(lines):
    angle_of_lines = {}
    for line in lines:
        assert line[2]-line[0] >= 0 , f"P1 must be to the right of P0"
        angle = np.arctan2(line[3]-line[1], line[2]-line[0])
        angle_of_lines[tuple(line)] = angle
    return angle_of_lines
    
def eliminate_unrelated_lines(lines, epsilon = 10*np.pi/180, n_neighbours=10):
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

def build_board(cells, img, digit_recognizer, show=True, crop_val_factor=0.05, blank_threshold=0.99):
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
            cv2.imshow("prediction: "+ str(val), img_digit)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        board[cell] = val
    
    return board

def extract_numbers_and_cells(img, show=False):
    img_sudoko_color = img
    res = extract_board(img)  
    if res is None:
        return None
    img_sudoko_color, M = res 
        
    img_sudoko = cv2.cvtColor(img_sudoko_color, cv2.COLOR_BGR2GRAY)

    img_edges = get_edges(img_sudoko)

    lines = get_lines(img_edges)
    if lines is None or len(lines) < 50:
        return None
    img_lines = draw_lines(np.copy(img_sudoko_color), lines)

    if show:
        img = combine_images([img_sudoko_color, img_edges, img_lines])
        cv2.imshow("board", img)
        cv2.waitKey(0)

    # Clean some lines
    lines = [construct_line(line[:2], line[2:]) for line in lines]
    lines = extend_lines(lines, x_max=img_sudoko_color.shape[1], y_max=img_sudoko_color.shape[0])
    lines = np.ceil(lines).astype('int')
    assert np.all(np.diff(np.array(lines)[:,(0,2)],axis=1) >= 0), f"P1 should be to the right of P0."

    if show:
        cv2.imshow("extended lines", draw_lines(np.copy(img_edges), lines))
        cv2.waitKey(0)
    lines = eliminate_unrelated_lines(lines)
    if show:
        cv2.imshow("removed unrelated lines", draw_lines(np.copy(img_edges), lines))
        cv2.waitKey(0)
    lines = eliminate_duplicate_lines(lines)
    if show:
        cv2.imshow("removed duplicated lines before warp", draw_lines(np.copy(img_edges), lines))
        cv2.waitKey(0)
    if len(lines) == 0:
        return None

    res = eliminate_duplicated_line_after_warp(lines)
    if res is None:
        return None
    h_lines, v_lines = res
    if show:
        cv2.imshow("removed duplicated lines after warp", draw_lines(np.copy(img_edges), h_lines+v_lines))
        cv2.waitKey(0)



    cells = extract_cells(v_lines, h_lines)
    good_cells = set()
    for i in range(9):
        for j in range(9):
            good_cells.add((i,j))
    for cell in cells:
        if not cell in good_cells:
            return None

    if show:
        cv2.destroyAllWindows()
    return cells, img_sudoko_color, img_lines

def extract_board(image):
    # Taken from https://stackoverflow.com/questions/57636399/how-to-detect-sudoku-grid-board-in-opencv

    def perspective_transform(image, corners):
        def order_corner_points(corners):
            # Separate corners into individual points
            # Index 0 - top-right
            #       1 - top-left
            #       2 - bottom-left
            #       3 - bottom-right
            if len(corners) < 4:
                return None
            corners = [(corner[0][0], corner[0][1]) for corner in corners]
            top_r, top_l, bottom_l, bottom_r = corners[0], corners[1], corners[2], corners[3]
            return (top_l, top_r, bottom_r, bottom_l)

        # Order points in clockwise order
        ordered_corners = order_corner_points(corners)
        if ordered_corners is None:
            return None
        top_l, top_r, bottom_r, bottom_l = ordered_corners

        # Move the corners outside to increase the square they envelop
        # increase_factor = 0.1
        # top_l = (top_l[0]*(1-increase_factor), top_l[1]*(1-increase_factor))
        # top_r = (top_r[0]*(1+increase_factor), top_r[1]*(1-increase_factor))
        # bottom_r = (bottom_r[0]*(1+increase_factor), bottom_r[1]*(1+increase_factor))
        # bottom_l = (bottom_l[0]*(1-increase_factor), bottom_l[1]*(1+increase_factor))
        top_l = (top_l[0]-10, top_l[1]-10)
        top_r = (top_r[0]+10, top_r[1]-10)
        bottom_r = (bottom_r[0]+10, bottom_r[1]+10)
        bottom_l = (bottom_l[0]-10, bottom_l[1]+10)
        ordered_corners = (top_l, top_r, bottom_r, bottom_l)



        # Determine width of new image which is the max distance between 
        # (bottom right and bottom left) or (top right and top left) x-coordinates
        width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
        width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
        width = max(int(width_A), int(width_B))

        # Determine height of new image which is the max distance between 
        # (top right and bottom right) or (top left and bottom left) y-coordinates
        height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
        height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
        height = max(int(height_A), int(height_B))

        # Construct new points to obtain top-down view of image in 
        # top_r, top_l, bottom_l, bottom_r order
        dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], 
                        [0, height - 1]], dtype = "float32")

        # Convert to Numpy format
        ordered_corners = np.array(ordered_corners, dtype="float32")



        # Find perspective transform matrix
        matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)

        # Return the transformed image
        return cv2.warpPerspective(image, matrix, (width, height)), matrix

    
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,3)

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        res = perspective_transform(original, approx)
        if res is None:
            return None
        transformed, M = res
        break

    return transformed, M


def read_cam(q:multiprocessing.Queue):
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        if ret:
            try:
                q.get_nowait()
            except Empty:
                pass
            q.put(img)
            
            cv2.imshow("Camera", img)
            cv2.waitKey(1)



if __name__ == '__main__':
    q = multiprocessing.Queue(maxsize=1)

    cam_reader = multiprocessing.Process(target=read_cam, args=(q, ))
    cam_reader.start()    

    digit_recognizer = load_model()
    while True:
        img = q.get()
        
        res = extract_numbers_and_cells(img, show=False)
        if res is not None:
            cells, img_warped, img_lines = res
            board = build_board(cells, img_warped, digit_recognizer, show=False)
            if board is None:
                continue
            try:
                solution = solve(board)
                if solution is None:
                    continue

                solved_board = create_board(640, 480, solution)
                combined = combine_images([img, solved_board])
                cv2.destroyAllWindows()
                cv2.imshow("Solution", combined)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
            except ValueError:
                print("Invalid board")
            
            
    
    cam_reader.kill()

