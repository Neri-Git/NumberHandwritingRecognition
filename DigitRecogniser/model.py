import tensorflow as tf
from tensorflow.keras import layers, models
from data import load_data

MODEL_PATH = "digit_model.keras"

def build_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_and_save():
    (x_train, y_train), (x_test, y_test) = load_data()

    model = build_model()
    model.fit(x_train, y_train, epochs=5, validation_split=0.1)

    model.save(MODEL_PATH)
    return model


def load_model():
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except:
        return train_and_save()