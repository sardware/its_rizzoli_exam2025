import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

#Percorso del dataset
dataset_path = "/Users/michelepotsios/Desktop/progetto rilevamento segni /"
train_csv = os.path.join(dataset_path, "sign_mnist_train.csv")
test_csv = os.path.join(dataset_path, "sign_mnist_test.csv")

# Verifica che i file esistano
if not os.path.exists(train_csv) or not os.path.exists(test_csv):
    raise FileNotFoundError(f" Il file {train_csv} o {test_csv} non esiste. Controlla il percorso!")

print("✅ Caricamento del dataset...")
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)
print("✅ Dataset caricato con successo.")

# Preparazione dei dati
X_train = train_df.drop("label", axis=1).values.reshape(-1, 28, 28, 1) / 255.0
y_train = tf.keras.utils.to_categorical(train_df["label"].values, num_classes=25)

# Suddivisione train/validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# ata Augmentation
datagen = ImageDataGenerator(
    rotation_range=15, width_shift_range=0.15, height_shift_range=0.15,
    zoom_range=0.2, shear_range=0.1, horizontal_flip=True
)
datagen.fit(X_train)

# Definizione del modello CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001), input_shape=(28, 28, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(25, activation="softmax")
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="categorical_crossentropy", metrics=["accuracy"])

# Early Stopping e Riduzione del Learning Rate
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5)

# Addestramento del modello
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=50,
    validation_data=(X_val, y_val),
    steps_per_epoch=len(X_train) // 64,
    callbacks=[early_stopping, reduce_lr]
)

#   Salvataggio del modello
model_save_path = os.path.join(dataset_path, "sign_language_model.h5")
model.save(model_save_path)
print(f"Modello salvato in {model_save_path}")