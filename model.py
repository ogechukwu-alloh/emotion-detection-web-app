"""
Keras CNN training script for 7-class emotion detection.
Saves best model to models/emotion_cnn.h5 (auto-loaded by app.py).
"""
import os, numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def build_model(input_shape=(48,48,1), num_classes=7):
    m = Sequential([
        Conv2D(32,(3,3),activation="relu",padding="same",input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32,(3,3),activation="relu",padding="same"),
        MaxPooling2D((2,2)),
        Dropout(0.25),
        Conv2D(64,(3,3),activation="relu",padding="same"),
        BatchNormalization(),
        Conv2D(64,(3,3),activation="relu",padding="same"),
        MaxPooling2D((2,2)),
        Dropout(0.25),
        Conv2D(128,(3,3),activation="relu",padding="same"),
        BatchNormalization(),
        Conv2D(128,(3,3),activation="relu",padding="same"),
        MaxPooling2D((2,2)),
        Dropout(0.25),
        Flatten(),
        Dense(256,activation="relu"),
        Dropout(0.5),
        Dense(num_classes,activation="softmax"),
    ])
    m.compile(optimizer=Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

def load_data():
    # TODO: Replace with real data (48x48 grayscale) for training.
    x_train = np.random.rand(512,48,48,1).astype("float32")
    y_train = np.random.randint(0,7,size=(512,)).astype("int64")
    x_val = np.random.rand(128,48,48,1).astype("float32")
    y_val = np.random.randint(0,7,size=(128,)).astype("int64")
    return (x_train,y_train),(x_val,y_val)

def main():
    (x_train,y_train),(x_val,y_val) = load_data()
    model = build_model()
    os.makedirs("models", exist_ok=True)
    ckpt = ModelCheckpoint("models/emotion_cnn.h5", monitor="val_accuracy", mode="max", save_best_only=True, verbose=1)
    early = EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
    model.fit(x_train,y_train, validation_data=(x_val,y_val), epochs=25, batch_size=64, callbacks=[ckpt,early])
    model.save("models/emotion_cnn.h5")
    print("Saved trained model to models/emotion_cnn.h5")

if __name__ == "__main__":
    main()
