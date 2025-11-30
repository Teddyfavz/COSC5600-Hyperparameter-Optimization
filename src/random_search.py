import time
import json
import random
import numpy as np
import tensorflow as tf
from cnn import CNNModel


class RandomSearch:
    """
    Implementation of the Random Search algorithm for hyperparameter optimization.
    """

    def __init__(self, data_handler, n_iter=10):
        self.data_handler = data_handler
        self.n_iter = n_iter

        self.results = []
        self.best_config = None
        self.best_score = -np.inf
        self.best_history = None
        self.total_search_time = 0

    # Generate random hyperparameters
    def sample_config(self):
        return {
            "lr": random.uniform(0.0001, 0.01),
            "batch_size": random.choice([8, 16, 32, 64, 128]),
            "epochs": random.choice(list(range(5, 51, 5)))
        }

    # Train a model for one configuration
    def run_trial(self, config):
        tf.keras.backend.clear_session()

        model = CNNModel(lr=config["lr"])
        model.build()
        model.compile()

        start = time.time()

        history = model.train(
            self.data_handler.x_train,
            self.data_handler.y_train,
            self.data_handler.x_val,
            self.data_handler.y_val,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            augment=False
        )

        duration = time.time() - start
        val_acc = max(history.history["val_accuracy"])

        return val_acc, duration, history.history

    # Full Random Search
    def search(self):
        print(f"Starting Random Search for {self.n_iter} trials...")
        search_start = time.time()

        for i in range(self.n_iter):
            print(f"\n=== Trial {i+1}/{self.n_iter} ===")

            config = self.sample_config()
            print("Config:", config)

            val_acc, duration, hist = self.run_trial(config)

            # Save result
            self.results.append({
                "config": config,
                "val_acc": val_acc,
                "train_time": duration,
                "history": hist
            })

            # Best model check
            if val_acc > self.best_score:
                self.best_score = val_acc
                self.best_config = config
                self.best_history = hist

        self.total_search_time = time.time() - search_start

        print("\nRandom Search Completed.")
        print(f"Best Validation Accuracy: {self.best_score:.4f}")
        print("Best Configuration:", self.best_config)
        print(f"Total Search Time: {self.total_search_time:.2f} seconds")

        return self.best_config, self.best_score

    # Save all results to JSON
    def save_results(self, filename="random_search_results.json"):
        payload = {
            "best_config": self.best_config,
            "best_score": float(self.best_score),
            "total_search_time": float(self.total_search_time),
            "results": self.results
        }

        with open(filename, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"Results saved to {filename}")

    # Return best model's accuracy history
    def get_best_history(self):
        if self.best_history is None:
            raise ValueError("No best history recorded.")
        return self.best_history