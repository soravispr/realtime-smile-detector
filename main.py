import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(1)
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
model = load_model('./keras_model/smiley.h5')

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)

    if len(faces) != 0:

        for (x, y, w, h) in faces:
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            face_roi = frame[y:y + h, x:x + w, :]
            face_roi = cv2.resize(face_roi, (64, 64))
            input_arr = img_to_array(face_roi).reshape(1,64,64,3)
            pred = model.predict(input_arr)

            if pred > 0.5:
                cv2.putText(img, 'Smile', (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                cv2.putText(img, 'Not smile', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow('camera', img)
    else:
        cv2.imshow('camera', frame)

    key = cv2.waitKey(1)
    if key % 256 == 27:
        break