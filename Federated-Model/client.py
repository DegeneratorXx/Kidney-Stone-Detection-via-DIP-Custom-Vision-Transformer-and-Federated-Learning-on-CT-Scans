import flwr as fl
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from model import Net
import numpy as np

class MnistClient(fl.client.NumPyClient):
    def __init__(self, model, train_data, test_data):
        self.model = model
        self.model.compile(
            optimizer=Adam(learning_rate=0.01),
            loss=SparseCategoricalCrossentropy(),
            metrics=[SparseCategoricalAccuracy()]
        )
        self.train_data = train_data
        self.test_data = test_data

    def get_parameters(self, config):
        return [val.numpy() for val in self.model.get_weights()]

    def set_parameters(self, parameters):
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.fit(self.train_data, epochs=1, verbose=1)
        return self.get_parameters({}), len(self.train_data), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.model.evaluate(self.test_data, verbose=0)
        return loss, accuracy, {}

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32).shuffle(10000)
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
    
    return train_data, test_data

def main():
    model = Net()
    train_data, test_data = load_data()
    client = MnistClient(model, train_data, test_data)
    fl.client.start_client(server_address="localhost:8080", client=client)

if __name__ == "__main__":
    main()

