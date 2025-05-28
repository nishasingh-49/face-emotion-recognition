import os
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# === CONFIG ===
BASE_MODEL_PATH = r"C:\Users\ir-vrl.helpdeskusb\Desktop\fr_2\facialemotionmodel.h5"  # Your existing model
TRAIN_DIR = r"C:\Users\ir-vrl.helpdeskusb\Desktop\fr_2\dataset\train"     # Your training images folder
VAL_DIR = r"C:\Users\ir-vrl.helpdeskusb\Desktop\fr_2\dataset\validation"   # Your validation images folder
FINETUNED_MODEL_PATH = r"C:\Users\ir-vrl.helpdeskusb\Desktop\fr_2\facialemotionmodel_finetuned.h5"

IMAGE_SIZE = (48, 48)  # Adjust if needed (usually 48x48 grayscale for emotion)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4

# Load your existing model
model = load_model(BASE_MODEL_PATH)

# Freeze all layers except last 2 layers to keep old knowledge
for layer in model.layers[:-2]:
    layer.trainable = False

# Compile model
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Data augmentation for training to improve generalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Callbacks to stop early if no improvement and save best model
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint(FINETUNED_MODEL_PATH, monitor='val_loss', save_best_only=True)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)

print(f"Fine-tuning complete! Model saved at {FINETUNED_MODEL_PATH}")
