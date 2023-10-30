import cv2

image = cv2.imread('Images/people1.jpg')
print(image.shape)

imagem = cv2.resize(image, (800, 600))

image_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
print(image_gray.shape)

face_detector = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
detections = face_detector.detectMultiScale(image_gray, scaleFactor = 1.09) #scaleFactor ajuste segun el tamano de la imagen y evitar falsos positivos.

print(detections)

print(len(detections))

for(x, y, w, h) in detections:
    #print(x, y,w, h)
    cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 255), 2)

cv2.imshow('WindowTest', imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()