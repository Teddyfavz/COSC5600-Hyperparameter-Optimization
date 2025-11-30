import os
import random
import numpy as np
import tensorflow as tf

from data import DataHandler
from cnn import CNNModel
from hyperband_search import HyperbandSearch

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["PYTHONHASHSEED"] = "42"

class MainCNN:
    def run(self):
        print("=== Loading and Preprocessing Data ===")
        data = DataHandler(flatten_labels=True, validation_size=0.2)
        data.load_and_preprocess()

        print("\n=== Building CNN Model ===")
        cnn = CNNModel(input_shape=(32, 32, 3), num_classes=10, lr=0.001)
        cnn.build()
        cnn.compile()

        print("\n=== Training Model ===")
        cnn.train(data.x_train, data.y_train, data.x_val, data.y_val,
                  epochs=10, batch_size=32)

        print("\n=== Evaluating Model ===")
        cnn.evaluate(data.x_test, data.y_test)
        cnn.plot_accuracy()
        cnn.predict_sample(data.x_test, data.y_test, index=5)


if __name__ == "__main__":
    MainCNN().run()