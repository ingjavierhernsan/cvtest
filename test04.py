import cv2

image = cv2.imread('Images/people2.jpg')
print(image.shape)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(image_gray.shape)

face_detector = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
detections = face_detector.detectMultiScale(image_gray, scaleFactor = 1.2, minNeighbors = 7, minSize = (10, 10), maxSize = (45, 45)) 

print(detections)

print(len(detections))

for(x, y, w, h) in detections:
    print(w, h)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

cv2.imshow('WindowTest', image)
cv2.waitKey(0)
cv2.destroyAllWindows()