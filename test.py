import cv2
import time

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, img = cap.read()
    
    if ret:
        # Display the resulting frame
        cv2.imshow('frame',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # cv2.imwrite("./img/sudoko.png", img)
            cv2.imwrite("./img/todo/"+str(time.time()) + ".png", img)
            break
        

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()









