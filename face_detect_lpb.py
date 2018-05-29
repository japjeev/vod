#import required libraries 
import numpy as np
import cv2

def detect_faces(f_cascade, colored_img, scaleFactor = 1.1):
    img_copy = np.copy(colored_img)
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    
    #let's detect multiscale (some images may be closer to camera than others) images
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);
    
    #go over list of faces and draw them as rectangles on original colored img
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)

    ## image with rectangles around faces
    #return img_copy

    # create a temp image and a mask to work on
    image = colored_img.copy()
    tempImg = colored_img.copy()
    maskShape = (image.shape[0], image.shape[1], 1)
    mask = np.full(maskShape, 0, dtype=np.uint8)

    # start the face loop
    i = 0
    for (x, y, w, h) in faces:
        i = i + 1
        #blur first so that the circle is not blurred
        tempImg [y:y+h, x:x+w] = cv2.blur(tempImg [y:y+h, x:x+w] ,(23,23))
        # create the circle in the mask and in the tempImg, notice the one in the mask is full
        cv2.circle(tempImg , ( int((x + x + w )/2), int((y + y + h)/2 )), int (h / 2), (0, 255, 0), 5)
        cv2.circle(mask , ( int((x + x + w )/2), int((y + y + h)/2 )), int (h / 2), (255), -1)
        cx = (2*x + w)/2
        cy = (2*y + h)/2
        text = str(i)
        cv2.putText(tempImg, text, (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # oustide of the loop, apply the mask and save
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(image,image,mask = mask_inv)
    img2_fg = cv2.bitwise_and(tempImg,tempImg,mask = mask)
    dst = cv2.add(img1_bg,img2_fg)

    ## image with blurred circles around faces
    return dst

#load cascade classifier training file for lbpcascade
lbp_face_cascade = cv2.CascadeClassifier('C:\opencv_3200\sources\data\lbpcascades\lbpcascade_frontalface.xml')

#load test image
test2 = cv2.imread('vods.jpg')
#call our function to detect faces
faces_detected_img = detect_faces(lbp_face_cascade, test2)

cv2.imshow('result', faces_detected_img)
cv2.imwrite("faces.jpg", faces_detected_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
