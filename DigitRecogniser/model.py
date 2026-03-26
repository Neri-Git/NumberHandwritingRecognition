import tensorflow as tf
from tensorflow.keras import layers, models
from data import load_data
from data import load_letters


DIGIT_MODEL_PATH = "digit_model.keras"
LETTER_MODEL_PATH = "letter_model.keras"

def build_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(36, activation='softmax')
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

    model.fit(
        x_train, y_train,
        epochs=10,
        validation_split=0.1
    )

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("Test accuracy:", test_acc)

    model.save(DIGIT_MODEL_PATH)
    return model


def load_model():
    try:
        return tf.keras.models.load_model(DIGIT_MODEL_PATH)
    except:
        return train_and_save()

def build_letter_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),

        layers.Dense(26, activation='softmax')  # A–Z
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_letters():
    (x_train, y_train), (x_test, y_test) = load_letters()

    model = build_letter_model()

    model.fit(
        x_train, y_train,
        epochs=10,
        validation_split=0.1
    )

    model.save(LETTER_MODEL_PATH)
    return model


def load_letter_model():
    try:
        return tf.keras.models.load_model(LETTER_MODEL_PATH)
    except:
        return train_letters()