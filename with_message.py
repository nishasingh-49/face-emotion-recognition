import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

model = load_model("facialemotionmodel_finetuned_v2.keras")
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_face(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 48, 48, 1))
    return reshaped

cap = cv2.VideoCapture(0)
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    elapsed = time.time() - start_time
    max_alpha = 255
    fade_duration = 2.0
    alpha = min(int((elapsed / fade_duration) * max_alpha), max_alpha)

    welcome_text = "Welcome to the Meeting, have a great day ahead!"
    (text_width, text_height), _ = cv2.getTextSize(welcome_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    frame_width = frame.shape[1]
    x_position = (frame_width - text_width) // 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (x_position - 10, 10), (x_position + text_width + 10, 10 + text_height + 10), (0, 0, 128), -1)
    cv2.addWeighted(overlay, alpha / 255.0, frame, 1 - alpha / 255.0, 0, frame)
    cv2.putText(frame, welcome_text, (x_position, 10 + text_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)                         
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            continue
        processed = preprocess_face(face_roi)
        prediction = model.predict(processed)
        emotion_index = np.argmax(prediction)
        emotion = emotion_labels[emotion_index]
        confidence = prediction[0][emotion_index] * 100

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
        label = f"{emotion} ({confidence:.1f}%)"
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (x, y - label_height - 10), (x + label_width, y), (0, 0, 128), -1)
        cv2.putText(frame, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Real-Time Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
