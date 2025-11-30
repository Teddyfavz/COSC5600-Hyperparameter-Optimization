import matplotlib.pyplot as plt
from data import DataHandler
from bayesian_search import BayesianSearch


# Plot accuracy curve for best Bayesian model
def plot_bayesian_accuracy(history):
    train_acc = history["accuracy"]
    val_acc = history["val_accuracy"]

    plt.figure(figsize=(8, 6))
    plt.plot(train_acc, label="Train Accuracy", color="red")
    plt.plot(val_acc, label="Validation Accuracy", color="green")

    plt.title("Training vs Validation Accuracy (Best Bayesian Optimization Model)")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main runner
class Main:
    def run(self):
        print("=== Loading CIFAR-10 Dataset ===")
        data = DataHandler(flatten_labels=True, validation_size=0.2)
        data.load_and_preprocess()

        print("\n=== Running Bayesian Optimization ===")
        bo = BayesianSearch(
            data_handler=data,
            n_init=5,
            n_iter=20
        )

        best_cfg, best_score = bo.search()
        bo.save_results("bayesian_search_results.json")

        print("\nBest Bayesian Config:", best_cfg)
        print("Best Validation Accuracy:", best_score)

        print("\n=== Plotting Best Accuracy Curve ===")
        best_hist = bo.get_best_history()
        plot_bayesian_accuracy(best_hist)

if __name__ == "__main__":
    Main().run()