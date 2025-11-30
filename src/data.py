import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np

class DataHandler:
    """
    Handles all dataset operations:
        -   Load data
        -   Normalize pixel value
        -   Flattens labels
        -   Splits training/validate data
        -   Visualizes sample images
    """
    def __init__(self, flatten_labels=True, validation_size = 0.2):
        self.flatten_labels = flatten_labels
        self.validation_size = validation_size
        self.x_train, self.x_val, self.y_train, self.y_val = None, None, None, None
        self.x_test, self.y_test = None, None

    def load_and_preprocess(self):
        """Load CIFAR-10 dataset and prepare training, validation and test sets."""
        print("Loading CIFAR-10 dataset....")
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        print(f"Dataset loaded successfully: {x_train.shape[0]} training images, {x_test.shape} test images.")

        #   Normalize pixel values to [0,1]
        x_train, x_test = x_train/255.0, x_test/255.0

        #   Flatten labels
        if self.flatten_labels:
            y_train, y_test = y_train.flatten(), y_test.flatten()

        #   Split training into training and validation set
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=self.validation_size, random_state=42
        )

        #   Store data in an object
        self.x_train, self.x_val, self.y_train, self.y_val = x_train, x_val, y_train, y_val
        self.x_test, self.y_test = x_test, y_test
        print(f"Shapes — Train: {x_train.shape}, Validation: {x_val.shape}, Test: {x_test.shape}")

    def visualize_samples(self, num_row =5, num_col = 5):
        if self.x_train is None:
            print("Data not loaded yet. Run load_and_preprocess() first")
            return

        fig, ax = plt.subplots(num_row, num_col, figsize = (8,8))
        k = 0
        for i in range(num_row):
            for j in range(num_col):
                ax[i][j].imshow(self.x_train[k])
                ax[i][j].axis('off')
                k+= 1
        plt.suptitle("Sample CIFAR-10 Images", fontsize=14)
        plt.show()
