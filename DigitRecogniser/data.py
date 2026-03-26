import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


def load_mnist():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = (x_train / 255.0).reshape(-1, 28, 28, 1)
    x_test = (x_test / 255.0).reshape(-1, 28, 28, 1)

    return (x_train, y_train), (x_test, y_test)


def load_emnist():
    ds_train, ds_test = tfds.load(
        'emnist/digits',
        split=['train', 'test'],
        as_supervised=True
    )

    def preprocess(image, label):
        # Fix rotation
        image = tf.transpose(image, perm=[1, 0, 2])

        # Normalize
        image = tf.cast(image, tf.float32) / 255.0

        return image, label

    ds_train = ds_train.map(preprocess).batch(60000)
    ds_test = ds_test.map(preprocess).batch(10000)

    x_train, y_train = next(iter(ds_train))
    x_test, y_test = next(iter(ds_test))

    return (x_train.numpy(), y_train.numpy()), (x_test.numpy(), y_test.numpy())


def load_data():
    (mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = load_mnist()
    (emnist_x_train, emnist_y_train), (emnist_x_test, emnist_y_test) = load_emnist()

    # Merge datasets
    x_train = np.concatenate([mnist_x_train, emnist_x_train])
    y_train = np.concatenate([mnist_y_train, emnist_y_train])

    x_test = np.concatenate([mnist_x_test, emnist_x_test])
    y_test = np.concatenate([mnist_y_test, emnist_y_test])

    print("Combined train shape:", x_train.shape)
    print("Combined test shape:", x_test.shape)

    return (x_train, y_train), (x_test, y_test)

def load_letters():
    import tensorflow_datasets as tfds
    import tensorflow as tf

    ds_train, ds_test = tfds.load(
        'emnist/letters',
        split=['train', 'test'],
        as_supervised=True
    )

    def preprocess(image, label):
        # Fix rotation
        image = tf.transpose(image, perm=[1, 0, 2])

        # Normalize
        image = tf.cast(image, tf.float32) / 255.0

        # Labels: 1–26 → 0–25
        label = label - 1

        return image, label

    ds_train = ds_train.map(preprocess).batch(100000)
    ds_test = ds_test.map(preprocess).batch(20000)

    x_train, y_train = next(iter(ds_train))
    x_test, y_test = next(iter(ds_test))

    return (x_train.numpy(), y_train.numpy()), (x_test.numpy(), y_test.numpy())