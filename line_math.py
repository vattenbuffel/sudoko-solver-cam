
import cv2
import numpy as np

def construct_line(p1, p2):
    if p1[0] < p2[0]:
        return [p1[0], p1[1], p2[0], p2[1]]
    return [p2[0], p2[1], p1[0], p1[1]]

def extend_lines(lines, x_max=2, y_max=2, x_min=0, y_min=0, img=None):
    new_lines = []
    for line in lines:
        extended_line = extend_line(line, x_max=x_max, y_max=y_max, x_min=x_min, y_min=y_min)
        if extended_line is not None:
            new_lines.append(extended_line)
            if img is not None:
                from main import draw_lines, combine_images
                import cv2
                before = draw_lines(np.copy(img), [line])
                after = draw_lines(np.copy(img), [extended_line.astype('int')])
                combined = combine_images([before, after])
                cv2.imshow("extended line", combined)
                cv2.waitKey(0)
                test = 5


    return new_lines

def extend_line(L, x_max=2, y_max=2, x_min=0, y_min=0):
    h0 = [x_min, y_min, x_max, y_min]
    h1 = [x_min, y_max, x_max, y_max]
    v0 = [x_min, y_min, x_min, y_max]
    v1 = [x_max, y_min, x_max, y_max]

    x_at_y_max = intersection_lines(L, h1)[0]
    if x_at_y_max is None:
        p0 = intersection_lines(L, v1)
    elif x_at_y_max <= x_max and x_at_y_max >= x_min:
        p0 = intersection_lines(L, h1)
    elif x_at_y_max > x_max:
        p0 = intersection_lines(L, v1)
    elif x_at_y_max < x_min:
        p0 = intersection_lines(L, v0)
    else:
        raise Exception("This should be mathematically impossible to happend.")
       
    if p0[0] is None:
        return None

    x_at_y_min = intersection_lines(L, h0)[0]
    if x_at_y_min is None:
        p1 = intersection_lines(L, v0)
    elif x_at_y_min >= x_min and x_at_y_min <= x_max:
        p1 = intersection_lines(L, h0)
    elif x_at_y_min > x_max:
        p1 = intersection_lines(L, v1)
    elif x_at_y_min < x_min:
        p1 = intersection_lines(L, v0)
    else:
        raise Exception("This should be mathematically impossible to happend.")
        
    
    if p1[0] is None:
        return None
    
    extended_line = construct_line(p0,p1)
    assert extended_line[2] - extended_line[0] >= 0, f"P1 should be to the left of P0."
    epsilon = 1e-3
    assert extended_line[0] + epsilon > x_min and extended_line[2] - epsilon < x_max, "The x-coordinate is outside of the image"
    assert extended_line[1] + epsilon > y_min and extended_line[3] - epsilon < y_max, "The y-coordinate is outside of the image"
    return np.array(extended_line)

def intersection_lines(L1, L2):
    x1 = L1[0]
    x2 = L1[2]
    x3 = L2[0]
    x4 = L2[2]
    y1 = L1[1]
    y2 = L1[3]
    y3 = L2[1]
    y4 = L2[3]

    denominator = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if denominator == 0:
        #This happens if they never intersect, I think
        return None, None

    t = (x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)
    t/=denominator
    # u = (x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)
    # u/=denominator

    px = x1 + t*(x2-x1)
    py = y1 + t*(y2-y1)
    return np.array([px, py])


if __name__ == '__main__':
    L1 = [0,0,1,1]
    L2 = [0,1,1,0]

    assert np.all(np.array([0.5, 0.5]) == intersection_lines(L1, L2))

    assert np.all(construct_line([0,0], [1,1]) ==  np.array([0,0,1,1]))
    assert np.all(construct_line([1,1], [0,0]) ==  np.array([0,0,1,1]))

    assert np.all(extend_line([0.5,0.5,1,1]) == np.array([0,0,2,2]))
    assert np.all(extend_line([0.5,0.5,0.5,1]) == np.array([0.5,0,0.5,2]))
    assert np.all(extend_line([0.5,0.5,1,0.5]) == np.array([0,0.5,2,0.5]))
    assert np.all(extend_line([0.5,0.5,1,1], x_max=100) == np.array([0,0,2,2]))

    # test
    assert extend_line([341,209,533,206], x_max=1280, y_max=720)[0] == 0 and extend_line([341,209,533,206], x_max=1280, y_max=720)[2] == 1280
    assert extend_line([295,535,682,548], x_max=1280, y_max=720)[0] == 0 and extend_line([295,535,682,548], x_max=1280, y_max=720)[2] == 1280
    extend_line([37, 638, 410, 731], x_max=1280, y_max=720)





