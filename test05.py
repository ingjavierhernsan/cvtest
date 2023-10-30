import cv2

image = cv2.imread('Images/people1.jpg')

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face_detector = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
detections = face_detector.detectMultiScale(image_gray, scaleFactor = 1.3, minSize = (30, 30)) 

eye_detector = cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')
eye_detections = eye_detector.detectMultiScale(image_gray, scaleFactor = 1.1, minNeighbors = 10, maxSize = (70, 70))

for(x, y, w, h) in detections:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

for(x, y, w, h) in eye_detections:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow('WindowTest', image)
cv2.waitKey(0)
cv2.destroyAllWindows()