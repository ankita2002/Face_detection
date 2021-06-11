import cv2
#OpenCv contains all ML algorithm
#For face mask detection there are around 6k test, each work on the single block
#We use classification approch to solve this problem
#We use cascade method which hepls us to solve the problem in real time
#Instead of running 6k tests , it only runs a few to make sure its a face, once detected it uses some more
import sys
#sys. command was showing an error hence imported it to solve the error

#get user supplied value
imagePath = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"

#Create haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)  #cascPath is from line 12

#Read the image
image = cv2.imread(imagePath) #imagePath is from line 11
gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

#Detect faces in image
faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (30, 30))
#flags = cv2.cv.CV_HAAR_SCALE_IMAGE )
#detectMultiscale is used for object detection
#since we are calling it for faceCascade, Thus face is been detected
#gray -> grayscale image
#ScaleFactor is to compensate the size of face 
   #for example someone is closer to camera so the face will seem big, someone might be far, this is used to compensate for the size of face
#minNeighbors defines how many objects are found before moving to next
#We use moving window for object detection

#The function returns list of rectangle where it feels a face it been detected
print("Found {0} faces!".format(len(faces)))

#draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x,y), (x+w , y+h), (0, 255, 0), 2)
#This function returns 4 values: the x and y location of the rectangle, and the rectangle’s width and height (w , h).
#We use these values to draw a rectangle using the built-in rectangle() function.

cv2.imshow("Face found", image)
cv2.waitKey(0)
#we display the image and wait for the user to press a key.

#add this in shell
#$ python FaceDetection2.py abba.png haarcascade_frontalface_default.xml