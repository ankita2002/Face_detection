import cv2

#load required trained XML classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#capture frame from camera
cap = cv2.VideoCapture(0)

while 1:
    #read frame from  camera
    ret , img = cap.read()
    #convert gray scale of each frames
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #detectsfaces of differenyt sizes in input image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        #to draw rectangle on face
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x +w]

        #detects eyes 
        eyes = eye_cascade.detectMultiScale(roi_gray)

        #to draw rectangle on eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 127, 255), 2) 
            #cv2.line(roi_color,(0,0),(1,2),0)

    #display image
    cv2.imshow('img', img)

    #wait for Esc Key to stop
    k = cv2.waitKey(30) & 0xff
    if k == 2:
        break
#colse the window
cap.release()

#deallocate any associated memory usage
cv2.destroyAllWindows()