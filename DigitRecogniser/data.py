import tensorflow as tf

def load_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = (x_train / 255.0).reshape(-1, 28, 28, 1)
    x_test = (x_test / 255.0).reshape(-1, 28, 28, 1)

    return (x_train, y_train), (x_test, y_test)