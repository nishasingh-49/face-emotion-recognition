import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# 1. Load the existing fine-tuned model
model_path = "facialemotionmodel_finetuned.h5"
model = load_model(model_path)

# 2. Set dataset directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
VAL_DIR = os.path.join(BASE_DIR, "dataset", "validation")

# 3. Create ImageDataGenerators with augmentations
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

# 4. Create data generators
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=64,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=64,
    class_mode='categorical',
    shuffle=False
)

# 5. Recompile the model (in case optimizer wasn't saved)
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 6. Fine-tune for more epochs
EPOCHS = 10
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# 7. Save the updated model
model.save("facialemotionmodel_finetuned_v2.h5")

print("✅ Model fine-tuned and saved as 'facialemotionmodel_finetuned_v2.h5'")
# After training:
model.save("facialemotionmodel_finetuned_v2.keras")
print("✅ Model saved as 'facialemotionmodel_finetuned_v2.keras'")
  