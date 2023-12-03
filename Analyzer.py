import cv2
from tensorflow.keras.models import load_model
import numpy as np

# 감정 레이블과 인덱스 매핑
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 얼굴 감정 분석 모델 로드
model = load_model('Emotion_little_vgg.h5')

# 웹캠 열기
cap = cv2.VideoCapture(0)

while True:
    # 프레임 읽기
    ret, frame = cap.read()

    # 프레임 크기 변경
    frame = cv2.resize(frame, (640, 480))

    # 얼굴 감지
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 얼굴이 감지되면 감정 분석 수행
    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (48, 48))
        face_img = face_img / 255.0
        face_img = np.expand_dims(face_img, axis=0)
        face_img = np.expand_dims(face_img, axis=-1)

        # 감정 분석 예측
        emotion_pred = model.predict(face_img)
        emotion_idx = np.argmax(emotion_pred)
        emotion_label = emotion_labels[emotion_idx]

        # 감정 텍스트 표시
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # 화면에 프레임 출력
    cv2.imshow('Emotion Analysis', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠과 창 닫기
cap.release()
cv2.destroyAllWindows()
