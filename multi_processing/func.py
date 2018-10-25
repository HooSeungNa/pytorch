import cv2


def face_detector(image):
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    img = cv2.imread(image,cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if(faces ==[]):
        print(image,"false")
    