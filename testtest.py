import cv2

cap = cv2.VideoCapture(0)

while(True):
    ret, img = cap.read()
    
    if ret:
        cv2.imshow('frame',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # cv2.imwrite("./img/sudoko.png", img)
            cv2.imwrite("./img/"+str(time.time()) + ".png", img)
            break
        

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()









