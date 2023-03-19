import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

overlay_img = cv2.imread('face.png', -1)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    ting = True
    for (x,y,w,h) in faces:
        x-=50
        y-=50
        w+=100
        h+=100

        try:
            overlay_resized = cv2.resize(overlay_img, (w,h))
            alpha_s = overlay_resized[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(0, 3):
                img[y:y+h, x:x+w, c] = (alpha_s * overlay_resized[:, :, c] +
                                        alpha_l * img[y:y+h, x:x+w, c])
            x1, y1, w1, h1 = x, y, w, h
            ting = False
        except:
            pass
    if ting:
        try:
            for c in range(0, 3):
                img[y1:y1+h1, x1:x1+w1, c] = (alpha_s * overlay_resized[:, :, c] +
                                        alpha_l * img[y1:y1+h1, x1:x1+w1, c])
        except:
            pass

    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
