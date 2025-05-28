import cv2
from deepface import DeepFace
import os
import time
import tempfile

# ===== CONFIGURATION =====
MODEL_PATH = r"C:\Users\ir-vrl.helpdeskusb\Desktop\fr_2\facialemotionmodel.h5"
DEBUG_MODE = True

# ===== INITIALIZATION =====
if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: Model file not found at {MODEL_PATH}")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

# Emotion color mapping
EMOTION_COLORS = {
    'angry': (0, 0, 255),
    'disgust': (0, 153, 0),
    'fear': (153, 0, 153),
    'happy': (0, 255, 255),
    'sad': (255, 0, 0),
    'surprise': (0, 255, 0),
    'neutral': (255, 255, 255)
}

# Initial state
current_emotion = "Initializing..."
last_results = {'emotion': {e: 0 for e in EMOTION_COLORS}}
frame_count = 0
analysis_interval = 5  # Process every 5 frames

# ===== MAIN LOOP =====
while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame capture error")
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        if frame_count % analysis_interval == 0:
            try:
                face_roi = frame[y:y + h, x:x + w]

                # Save face to temporary file
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                    temp_filename = tmp_file.name
                    cv2.imwrite(temp_filename, face_roi)

                # Run analysis using file path
                last_results = DeepFace.analyze(
                    img_path=temp_filename,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=not DEBUG_MODE,
                    model_path=MODEL_PATH
                )

                # Delete temp file
                os.remove(temp_filename)

                # Normalize result format
                if isinstance(last_results, list):
                    last_results = last_results[0]

                current_emotion = last_results.get('dominant_emotion', 'Unknown')

                if DEBUG_MODE:
                    print(f"🎭 Detected Emotion: {current_emotion}")
                    print(f"📊 Emotions: {last_results.get('emotion', {})}")

            except Exception as e:
                if DEBUG_MODE:
                    print(f"❌ Analysis error: {str(e)}")
                current_emotion = "Error"

        # UI Drawing
        color = EMOTION_COLORS.get(current_emotion.lower(), (0, 255, 0))
        confidence = 0
        try:
            emotions = last_results.get('emotion', {})
            confidence = float(emotions.get(current_emotion, 0))
        except Exception:
            pass

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, current_emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Confidence: {confidence:.1f}%",
                    (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Optional FPS
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)

    frame_count += 1
    cv2.imshow('Emotion Detection - Press Q to Quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===== CLEANUP =====
cap.release()
cv2.destroyAllWindows()
