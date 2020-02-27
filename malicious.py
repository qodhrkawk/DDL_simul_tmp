from flmodel import Flmodel
from utils import MetaFunc, cal_weights_hash
import tensorflow as tf
import numpy as np


class Malicious:
    def __init__(
            self,
            id: str,  # unique ID (Address)
            flmodel: Flmodel = None,  # NN model
            x_train=None, y_train=None,
            x_test=None, y_test=None,
            reputation: dict = dict(),
            policy_update_reputation_name: str = None,
            policy_update_reputation_func: callable = None,
            policy_update_model_weights_name: str = None,
            policy_update_model_weights_func: callable = None,
            policy_replace_model_name: str = None,
            policy_replace_model_func: callable = None):
        self.id = id
        self.__flmodel = flmodel
        self.__x_train, self.__y_train = x_train, y_train
        self.__x_test, self.__y_test = x_test, y_test
        self.__reputation = reputation
        self.__policy_update_reputation = MetaFunc(
            policy_update_reputation_name,
            policy_update_reputation_func
        )
        self.__policy_update_model_weights = MetaFunc(
            policy_update_model_weights_name,
            policy_update_model_weights_func
        )
        self.__policy_replace_model = MetaFunc(
            policy_replace_model_name,
            policy_replace_model_func
        )

    def print(self):
        print("")
        print("id        :\t", self.id)
        print("weight    :\t", cal_weights_hash(self.get_model_weights()))
        print("train     :\t", self.get_train_size())
        print("test      :\t", self.get_test_size())
        print("reputation:\t", self.get_reputation())
        print("policies  :\t", "update_reputation: ",
              self.get_policy_update_reputation_name())
        print("policies  :\t", "update_model_weights: ",
              self.get_policy_update_model_weights_name())
        print("policies  :\t", "replace_model: ",
              self.get_policy_replace_model_name())

    """reputation"""

    def set_reputation(self, reputation: dict):
        self.__reputation = reputation  # (id: str => amount: float)

    def get_reputation(self):
        return self.__reputation

    """data"""

    def set_train_data(self, x_train, y_train):
        self.__x_train, self.__y_train = x_train, y_train

    def set_test_data(self, x_test, y_test):
        self.__x_test, self.__y_test = x_test, y_test

    def get_train_data(self):
        return self.__x_train, self.__y_train

    def get_test_data(self):
        return self.__x_test, self.__y_test

    def get_train_size(self):
        return len(self.__x_train) if self.__x_train is not None else 0

    def get_test_size(self):
        return len(self.__x_test) if self.__x_test is not None else 0

    """Flmodel"""

    def set_model(self, flmodel: Flmodel):
        self.__flmodel = flmodel

    def get_model(self):
        return self.__flmodel

    def fit_model(self, epochs=1, callbacks=[], verbose=0):
        self.__flmodel.fit(
            self.__x_train, self.__y_train,
            epochs=epochs, callbacks=callbacks, verbose=verbose)

    def evaluate_model(self, verbose=0):
        return self.__flmodel.evaluate(
            self.__x_test, self.__y_test, verbose=verbose)

    def get_model_metrics(self):
        return self.__flmodel.get_metrics()

    def get_model_weights(self):
        return self.__flmodel.get_weights()

    def set_model_weights(self, new_weights):
        self.__flmodel.set_weights(new_weights)

    def predict_model(self, x_input):
        predictions = self.__flmodel.predict(x_input)
        print(len(predictions))
        for i in range(len(predictions)):
            if np.argmax(predictions[i]) == 9:
                tmp = predictions[i][9]
                predictions[i][9] = predictions[i][2]
                predictions[i][2] = tmp
        return predictions

    """policies"""

    def update_reputation(self, *args):
        return self.__policy_update_reputation.func(*args)

    def update_model_weights(self, *args):
        return self.__policy_update_model_weights.func(*args)

    def replace_model(self, *args):
        return self.__policy_replace_model.func(*args)

    def get_policy_update_reputation_name(self):
        return self.__policy_update_reputation.name

    def get_policy_update_model_weights_name(self):
        return self.__policy_update_model_weights.name

    def get_policy_replace_model_name(self):
        return self.__policy_replace_model.name

    def set_policy_update_reputation(
            self,
            policy_update_reputation_name,
            policy_update_reputation_func):
        self.__policy_update_reputation = MetaFunc(
            policy_update_reputation_name,
            policy_update_reputation_func
        )

    def set_policy_update_model_weights(
            self,
            policy_update_model_weights_name,
            policy_update_model_weights_func):
        self.__policy_update_model_weights = MetaFunc(
            policy_update_model_weights_name,
            policy_update_model_weights_func
        )

    def set_policy_replace_model(
            self,
            policy_replace_model_name,
            policy_replace_model_func):
        self.__policy_replace_model = MetaFunc(
            policy_replace_model_name,
            policy_replace_model_func
        )


if __name__ == "__main__":
    import tensorflow as tf
    import numpy as np
    import arguments
    import time

    def create_model():
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
        return Flmodel(mnist_model)

    def my_policy_update_model_weights(self: Malicious, peer_weights: dict):
        # get reputation
        reputation = self.get_reputation()
        if len(reputation) == 0:
            raise ValueError
        if len(reputation) != len(peer_weights):
            raise ValueError

        ids = list(reputation.keys())
        total_reputation = sum(reputation.values())

        # set zero-filled NN layers
        new_weights = list()
        for layer in peer_weights[ids[0]]:
            new_weights.append(np.zeros(layer.shape))

        # calculate new_weights
        for i, layer in enumerate(new_weights):
            for id in ids:
                layer += peer_weights[id][i] * \
                    reputation[id] / total_reputation

        # set new_weights
        self.set_model_weights(new_weights)

    def equally_fully_connected(my_id: str, ids: list):
        reputation = dict()
        for id in ids:
            if id == my_id:
                continue
            reputation[id] = 1.
        return reputation

    def avg_time(times):
        if len(times) == 0:
            return 0.0
        else:
            return sum(times) / len(times)
