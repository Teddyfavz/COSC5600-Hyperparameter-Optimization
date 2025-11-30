
import tensorflow as tf
from scipy.stats import triang
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import time

from tensorflow.python.keras.saving.saved_model.serialized_attributes import metrics


class CNNModel:
    def __init__(self, input_shape=(32,32,3), num_classes=10,lr=0.001):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lr = lr
        self.model = None
        self.history = None

    def build(self):
        """Build the CNN architecture (Functional API)"""
        inputs = Input(shape = self.input_shape)

        #   Block 1 (Edges)
        x = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(inputs)
        x = BatchNormalization()(x)
        x = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2,2))(x)

        #   Block 2 (Shapes)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)

        #   Block 3 (Objects i.e cat, airplane etc)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)

        # Fully Connected Layers - Decision
        x = Flatten()(x)
        x = Dropout(0.2)(x)
        x = Dense (1024, activation = 'relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(self.num_classes, activation = 'softmax')(x)

        self.model = Model(inputs, outputs)
        print("CNN Model built successfully")
        return self.model

    def compile(self):
        optimizer = Adam(learning_rate=self.lr)
        self.model.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

    def train(self, x_train, y_train, x_val, y_val, epochs = 10, batch_size = 32, augment = False):
        start_time = time.time()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
        ]

        if augment:
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                width_shift_range = 0.1,
                height_shift_range = 0.1,
                horizontal_flip = True
            )
            train_gen = datagen.flow(x_train, y_train, batch_size = batch_size)
            step_per_epoch = x_train.shape[0] // batch_size
            self.history = self.model.fit(
                train_gen,
                validation_data = (x_val, y_val),
                steps_per_epoch = steps_per_epoch,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        else:
            self.history = self.model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )

        end_time = time.time()
        print(f"Training completed in {(end_time - start_time):.2f} seconds.")
        return self.history

    def evaluate(self, x_test, y_test):
        test_loss, test_acc = self.model.evaluate(x_test, y_test)
        print(f"Test Accuracy: {test_acc:.4f}")
        return test_loss, test_acc

    def plot_accuracy(self):
        plt.plot(self.history.history['accuracy'], label='Train Accuracy', color='red')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy', color='green')
        plt.title('Training vs Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    def predict_sample(self, x_test, y_test, index=0):
        """Predict and show one image from test data."""
        labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']

        plt.imshow(x_test[index])
        plt.title("Test Image")
        plt.show()

        prediction = self.model.predict(np.expand_dims(x_test[index], axis=0))
        predicted_label = labels[np.argmax(prediction)]
        true_label = labels[y_test[index]]

        print(f"True Label: {true_label} | Predicted Label: {predicted_label}")
