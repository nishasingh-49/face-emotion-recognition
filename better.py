import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = r"C:\Users\ir-vrl.helpdeskusb\Desktop\fr_2\facialemotionmodel.h5"
model = load_model(MODEL_PATH)

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

COLOR_MAP = {
    'Angry': (0, 0, 255),
    'Disgust': (0, 153, 0),
    'Fear': (153, 0, 153),
    'Happy': (0, 255, 255),
    'Sad': (255, 0, 0),
    'Surprise': (0, 255, 0),
    'Neutral': (200, 200, 200)
}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_face(image):
    """Convert face to grayscale, resize, normalize, and reshape."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(gray, (48, 48))
    normalized = face_resized / 255.0
    return normalized.reshape(1, 48, 48, 1)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access webcam.")
    exit()

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        if face.size == 0:
            continue

        input_face = preprocess_face(face)
        predictions = model.predict(input_face, verbose=0)[0]
        idx = int(np.argmax(predictions))
        emotion = EMOTIONS[idx]
        confidence = predictions[idx]

        label = f"{emotion} ({confidence * 100:.1f}%)"
        color = COLOR_MAP.get(emotion, (255, 255, 255))

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
