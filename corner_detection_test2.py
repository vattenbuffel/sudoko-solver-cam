import cv2
import numpy as np

def extract_board(image):

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


if __name__ == '__main__':

    image = cv2.imread("./img/sudoko5.png")
    transformed = extract_board(image)

    cv2.imshow('transformed', transformed)
    # cv2.imwrite('board.png', transformed)
    cv2.waitKey()