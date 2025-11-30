import time
import json
import numpy as np
import tensorflow as tf
from scipy.stats import norm
from cnn import CNNModel


class BayesianSearch:
    """
    Implementation of the Bayesian Search algorithm for hyperparameter optimization.
    """
    def __init__(self, data_handler, n_init=5, n_iter=20):
        self.data_handler = data_handler
        self.n_init = n_init
        self.n_iter = n_iter

        self.X = []
        self.y = []

        self.best_config = None
        self.best_score = -np.inf
        self.best_history = None
        self.results = []
        self.total_time = 0

        self.noise = 1e-6   # Gaussian Process noise

        # allowable epoch choice
        self.epoch_choices = np.array(list(range(5, 51, 5)))



    # Sample random hyperparameters
    def sample_random_config(self):
        """Sample a configuration from the hyperparameter search space."""
        return {
            "lr": float(10 ** np.random.uniform(-4, -2)),
            "batch_size": int(np.random.choice([8, 16, 32, 64, 128])),
            "epochs": int(np.random.choice(self.epoch_choices))
        }

    # Convert config to numeric vector for GP
    def config_to_vector(self, cfg):
        return np.array([
            np.log10(cfg["lr"]),
            np.log(cfg["batch_size"]),
            cfg["epochs"] / 50.0
        ])

    # Convert vector → config
    def vector_to_config(self, vec):
        # snap epoch to nearest valid choice
        epoch_norm = vec[2] * 50
        idx = np.argmin(np.abs(self.epoch_choices - epoch_norm))
        return {
            "lr": float(10 ** vec[0]),
            "batch_size": int(np.exp(vec[1])),
            "epochs": int(self.epoch_choices[idx])
        }


    # Train model using a config
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

        # save result
        self.results.append({
            "config": config,
            "val_acc": float(val_acc),
            "train_time": float(duration),
            "history": history.history
        })

        # update best model
        if val_acc > self.best_score:
            self.best_score = val_acc
            self.best_config = config.copy()
            self.best_history = history.history

        return val_acc


    # Gaussian Process Kernel
    def rbf_kernel(self, A, B, length_scale=1.0, sigma_f=1.0):
        A = np.atleast_2d(A)
        B = np.atleast_2d(B)
        sqdist = (
            np.sum(A**2, axis=1).reshape(-1, 1)
            + np.sum(B**2, axis=1)
            - 2 * np.dot(A, B.T)
        )
        return sigma_f**2 * np.exp(-0.5 * sqdist / length_scale**2)

    # Expected Improvement
    def expected_improvement(self, X_new, xi=0.01):
        if len(self.X) == 0:
            return 1e6

        X_train = np.array(self.X)
        y_train = np.array(self.y)

        K = self.rbf_kernel(X_train, X_train) + self.noise * np.eye(len(X_train))
        K_s = self.rbf_kernel(X_train, X_new)
        K_ss = self.rbf_kernel(X_new, X_new) + self.noise

        K_inv = np.linalg.inv(K)

        mu = K_s.T.dot(K_inv).dot(y_train)
        sigma = np.sqrt(np.maximum(K_ss - K_s.T.dot(K_inv).dot(K_s), 1e-9))

        mu_best = np.max(y_train)

        # Expected Improvement
        imp = mu - mu_best - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

        return float(ei)


    # Select next hyperparameters using EI
    def propose_next_config(self, n_candidates=50):
        candidates = []
        scores = []

        for _ in range(n_candidates):
            cfg = self.sample_random_config()
            vec = self.config_to_vector(cfg)
            candidates.append(vec)
            scores.append(self.expected_improvement(vec.reshape(1, -1)))

        best_idx = int(np.argmax(scores))
        return self.vector_to_config(candidates[best_idx])

    # Full Bayesian Optimization loop
    def search(self, verbose=True):
        print("Starting Bayesian Optimization...")
        start = time.time()

        # --- INITIAL RANDOM TRIALS ---
        for i in range(self.n_init):
            cfg = self.sample_random_config()
            if verbose:
                print(f"Initial sample {i+1}: {cfg}")

            val_acc = self.run_trial(cfg)
            self.X.append(self.config_to_vector(cfg))
            self.y.append(val_acc)

        #  Bayesian Optimization Iterations
        for i in range(self.n_iter):
            cfg = self.propose_next_config()

            if verbose:
                print(f"\n[BO Iter {i+1}] Trying config: {cfg}")

            val_acc = self.run_trial(cfg)
            self.X.append(self.config_to_vector(cfg))
            self.y.append(val_acc)

            if verbose:
                print(f" → val_acc={val_acc:.4f} | best={self.best_score:.4f}")

        self.total_time = time.time() - start

        print("\nBayesian Optimization Completed.")
        print("Best Config:", self.best_config)
        print("Best Score:", self.best_score)
        print("Total Time:", self.total_time)

        return self.best_config, self.best_score

    # Save Results to JSON
    def save_results(self, filename="bayesian_search_results.json"):
        payload = {
            "best_config": self.best_config,
            "best_score": float(self.best_score),
            "total_time": float(self.total_time),
            "results": self.results
        }

        with open(filename, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"Results saved to {filename}")


    # Return the history for best model
    def get_best_history(self):
        if self.best_history is None:
            raise ValueError("No best model history found.")
        return self.best_history