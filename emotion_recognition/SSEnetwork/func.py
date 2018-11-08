import cv2


def face_detector(image):
    save_dir='D:/emotion_dataset/mmidataset/VideoWithImageLabels/face_detected/'
    image_dir='D:/emotion_dataset/mmidataset/VideoWithImageLabels/image_dataset/'
    load=image_dir+image

    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    img = cv2.imread(load,cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_color = img[y:y+h, x:x+w]
        roi_color=cv2.resize(roi_color,(128,128),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(save_dir+image,roi_color)
        print(save_dir+image)