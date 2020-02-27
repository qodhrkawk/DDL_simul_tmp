import tensorflow as tf
import numpy as np


class Flmodel:
    def __init__(self, compiled_model):
        self.model = compiled_model
        self.__metrics = None  # loss, acc, et al.

    def fit(self, x_train, y_train, epochs=1, callbacks=[], verbose=0):
        self.model.fit(
            x_train, y_train,
            epochs=epochs, callbacks=callbacks, verbose=verbose)






    def evaluate(self, x_test, y_test, verbose=0):
        self.__metrics = self.model.evaluate(x_test, y_test, verbose=verbose)
        return self.get_metrics()

    def get_metrics(self):
        return self.__metrics

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, new_weights):
        self.model.set_weights(new_weights)

    def predict(self, x_input):
        return self.model.predict(x_input)

    def mal_predict(self, x_input):
        mal_x = self.model.predict(x_input)
        mal_x.fill(0)
        return mal_x



if __name__ == "__main__":
    # Load dataset
    mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Create model
    mnist_model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    mnist_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    flmodel = Flmodel(mnist_model)
    # weights = flmodel.get_weights()
    # print(weights)

    # Train
    flmodel.fit(x_train, y_train, epochs=5, verbose=2)

    # Eval.
    print("Eval: ", flmodel.evaluate(x_test, y_test, verbose=2))