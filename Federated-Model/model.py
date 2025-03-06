import tensorflow as tf
from tensorflow.keras import layers, models

class Net(tf.keras.Model):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu')
        self.pool = layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout1 = layers.Dropout(0.25)
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(128, activation='relu')
        self.dropout2 = layers.Dropout(0.5)
        self.fc2 = layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        return self.fc2(x)

# Test model
if __name__ == "__main__":
    model = Net()
    model.build((None, 28, 28, 1))
    model.summary()
