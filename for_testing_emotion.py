import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the fine-tuned model
model_path = os.path.join(os.getcwd(), "facialemotionmodel_finetuned.h5")
emotion_model = load_model(model_path)

# Define emotion labels (make sure this matches your model’s output order)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Preprocessing function
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or can't be opened")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 48, 48, 1))  # Shape for model input
    return reshaped, img

# Prediction function
def predict_emotion(image_path):
    try:
        processed_img, original_img = preprocess_image(image_path)
        prediction = emotion_model.predict(processed_img)
        max_index = np.argmax(prediction[0])
        emotion = emotion_labels[max_index]
        confidence = prediction[0][max_index] * 100

        print(f"\nPredicted Emotion: {emotion} ({confidence:.2f}%)")

        # Draw text on image
        cv2.putText(original_img, f"{emotion} ({confidence:.1f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Prediction", original_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {str(e)}")

# Test image path (update as needed)
sample_image_path = os.path.join("dataset", "train", "sad", "281.jpg")
if os.path.exists(sample_image_path):
    predict_emotion(sample_image_path)
else:
    print(f"Sample image not found at: {sample_image_path}")
