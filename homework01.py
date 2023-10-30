import cv2

image = cv2.imread('Images/car.jpg')

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

car_detector = cv2.CascadeClassifier('Cascades/cars.xml')
car_detection = car_detector.detectMultiScale(image_gray, scaleFactor = 1.03, minNeighbors = 1, minSize = (10, 10), maxSize = (150, 150))

for(x, y, w, h) in car_detection:
    print(w, h)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('WindowsTest', image)
cv2.waitKey(0)
cv2.destroyAllWindows()