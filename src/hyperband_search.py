import time
import json
from tabnanny import verbose
from turtledemo.penrose import start

import numpy as np
import self
import tensorflow as tf
from cnn import CNNModel

class HyperbandSearch:
    """
    Implementation of the Hyperband Search algorithm for hyperparameter optimization.
    """
    def __init__(self, max_iter = 50, eta =3, data_handler = None):
        self.max_iter = max_iter
        self.eta = eta
        self.data_handler = data_handler

        self.results = []
        self.best_config = None
        self.best_score = -np.inf
        self.best_history = None

        self.s_max = int(np.floor(np.log(self.max_iter)/ np.log(self.eta)))
        self.total_search_time = 0

    # Generating random hyperparameter configuration
    def get_random_config(self):
        return {
            "lr": float(10 ** np.random.uniform(np.log10(0.0001), np.log10(0.01))),
            "batch_size": int(np.random.choice([8, 16, 32, 64, 128])),
            "epochs": int(np.random.choice(range(5, 51, 5)))
        }

    # Training the model to build and return validation accuracy for a given configuration
    def run_trial(self, config, epochs):
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
            epochs=epochs,
            batch_size= config["batch_size"],
            augment= False
        )

        duration = time.time() - start
        val_acc = max(history.history["val_accuracy"])

        return val_acc, duration, history.history

    # Successive halving - Training and configuration for r epoch, keep top n/eta configuration
    def successive_halving(self, configs, n, r):
        results = []

        for i, cfg in enumerate(configs[:n]):
            print (f" Training config {i + 1}/{n} for {r} epochs....")

            try:
                val_acc, dur, hist = self.run_trial(cfg, r)
                results.append((cfg, val_acc, dur, hist))

                self.results.append({
                    "config": cfg,
                    "val_acc": val_acc,
                    "epochs": r,
                    "train_time": dur,
                    "history": hist
                })

                if val_acc > self.best_score:
                    self.best_score = val_acc
                    self.best_config = cfg.copy()
                    self.best_history = hist

            except Exception as e:
                print(f" Error with config {i + 1}: {e}")
                results.append((cfg, -1.0, 0, None))

        #sort and select the best performer
        results.sort(key=lambda x: x[1], reverse=True)
        keep_n = max(1, n // self.eta)
        kept = [cfg for cfg, _, _, _ in results[:keep_n]]

        return kept

    # Main Hyperband loop
    def search(self, verbose=True):
        """Run full Hyperband optimization search."""
        if self.data_handler is None:
            raise ValueError("DataHandler instance required for search.")

        start_time = time.time()
        print(f"Starting Hyperband Search | max_iter={self.max_iter}, eta={self.eta}, s_max={self.s_max}")

        for s in reversed(range(self.s_max + 1)):
            n = int(np.ceil((self.s_max + 1) * (self.eta ** s) / (s + 1)))
            r = self.max_iter * (self.eta ** (-s))

            if verbose:
                print(f"\nBracket s={s}: {n} configs | initial r={r} epochs")

            # Generate initial configs
            configs = [self.get_random_config() for _ in range(n)]

            # Successive Halving within each bracket
            for i in range(s + 1):
                n_i = int(n * (self.eta ** (-i)))
                r_i = max(5, min(self.max_iter, int(r * (self.eta ** i))))

                if verbose:
                    print(f" Round {i + 1}: {n_i} configs × {r_i} epochs")

                configs = self.successive_halving(configs, n_i, r_i)

        self.total_search_time = time.time() - start_time

        print("\nHyperband Search Completed.")
        print(f"Best Validation Accuracy: {self.best_score:.4f}")
        print(f"Best Configuration: {self.best_config}")
        print(f"Total Search Time: {self.total_search_time:.2f} seconds")

        return self.best_config, self.best_score

    # saving results to file
    def save_results(self, filename="hyperband_results.json"):
        payload = {
            "best_config": self.best_config,
            "best_score": float(self.best_score),
            "total_search_time": float(self.total_search_time),
            "results": self.results,
        }
        with open(filename, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Results saved to {filename}")

    # Get training curve (train & val accuracy) of Best Model
    def get_best_history(self):
        if self.best_history is None:
            raise ValueError("Best model history not available.")
        return self.best_history


