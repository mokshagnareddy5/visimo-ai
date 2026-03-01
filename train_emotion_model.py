"""
Train a LIGHTWEIGHT Facial Emotion Recognition model using FER2013 dataset.
Optimized for CPU training (~5-10 minutes).

Usage: python train_emotion_model.py
Output: emotion_model_v2.hdf5
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, Flatten,
    BatchNormalization, Input
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

print("=" * 50)
print("  LIGHTWEIGHT Emotion Model Training")
print("  Optimized for CPU (~5-10 min)")
print("=" * 50)

# ================================
# 1. LOAD DATA FROM FOLDERS
# ================================
data_dir = Path("fer2013")
train_dir = data_dir / "train"
test_dir = data_dir / "test"

if not train_dir.exists():
    print("ERROR: fer2013/train folder not found!")
    print("Run the previous script to download it first.")
    exit(1)

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
display_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
IMG_SIZE = 48  # Use 48x48 (native FER2013 size — faster training)

def load_images(base_dir):
    images, labels = [], []
    for idx, emotion in enumerate(emotion_labels):
        edir = base_dir / emotion
        if not edir.exists():
            continue
        count = 0
        for f in edir.iterdir():
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    images.append(img)
                    labels.append(idx)
                    count += 1
        print(f"  {emotion:>10}: {count}")
    return np.array(images), np.array(labels)

print("\n[1/4] Loading images...")
X_train, y_train = load_images(train_dir)
print(f"\n  Testing set:")
X_test, y_test = load_images(test_dir)

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

y_train_cat = tf.keras.utils.to_categorical(y_train, 7)
y_test_cat = tf.keras.utils.to_categorical(y_test, 7)

print(f"\n  Train: {X_train.shape}, Test: {X_test.shape}")

# ================================
# 2. DATA AUGMENTATION
# ================================
print("\n[2/4] Data augmentation...")
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
datagen.fit(X_train)

# ================================
# 3. LIGHTWEIGHT MODEL (Fast on CPU)
# ================================
print("\n[3/4] Building lightweight model...")

model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

    # Block 1 — Small filters
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    # Block 2
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    # Block 3
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.3),

    # Classifier
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(7, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"  Parameters: {model.count_params():,}")

# ================================
# 4. TRAIN
# ================================
print("\n[4/4] Training... (~5-10 min on CPU)")
print("       Best model is auto-saved. Press Ctrl+C to stop early.\n")

callbacks = [
    ModelCheckpoint('emotion_model_v2.hdf5', monitor='val_accuracy',
                    save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=8,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=3, min_lr=1e-6, verbose=1)
]

history = model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=128),  # Big batch = faster
    steps_per_epoch=len(X_train) // 128,
    epochs=30,
    validation_data=(X_test, y_test_cat),
    callbacks=callbacks,
    verbose=1
)

# ================================
# RESULTS
# ================================
print("\n" + "=" * 50)
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"  Test Accuracy: {test_acc*100:.2f}%")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
print("\n  Per-class accuracy:")
for i, label in enumerate(display_labels):
    mask = y_test == i
    if mask.sum() > 0:
        print(f"    {label:>10}: {(y_pred[mask] == i).mean()*100:.1f}%")

print(f"\n  Saved: emotion_model_v2.hdf5")
print(f"  Input shape: {IMG_SIZE}x{IMG_SIZE} grayscale")
print("=" * 50)
