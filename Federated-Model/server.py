import flwr as fl
import tensorflow as tf
from model import Net
import numpy as np

def get_on_fit_config(server_round):
    return {"server_round": server_round, "local_epochs": 1}

def get_evaluate_fn(model):
    def evaluate(server_round, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        return loss, {"accuracy": accuracy, "server_round": server_round}
    return evaluate

def main():
    global x_test, y_test
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test / 255.0
    x_test = x_test[..., np.newaxis]
    
    model = Net()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1,
        fraction_evaluate=0.1,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        on_fit_config_fn=get_on_fit_config,
        evaluate_fn=get_evaluate_fn(model)
    )

    fl.server.start_server(
        server_address="localhost:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=10)
    )

if __name__ == "__main__":
    main()
