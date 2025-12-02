
from data import DataHandler
from hyperband_search import HyperbandSearch
import matplotlib.pyplot as plt


# Plot function
def plot_hyperband_accuracy(history):
    train_acc = history["accuracy"]
    val_acc = history["val_accuracy"]

    plt.figure(figsize=(8, 6))
    plt.plot(train_acc, label="Train Accuracy", color="red")
    plt.plot(val_acc, label="Validation Accuracy", color="green")

    plt.title("Training vs Validation Accuracy (Best Hyperband Model)")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main Runner Class
class Main:
    def run(self):
        print("=== Loading & Preprocessing CIFAR-10 Dataset ===")
        data = DataHandler(flatten_labels=True, validation_size=0.2)
        data.load_and_preprocess()

        print("\n=== Starting Hyperband Hyperparameter Search ===")
        hb = HyperbandSearch(
            max_iter=50,
            eta=3,
            data_handler=data
        )

        best_config, best_score = hb.search()
        hb.save_results("hyperband_results.json")

        print("\nBest Hyperband Configuration:", best_config)
        print("Best Validation Accuracy:", best_score)

        print("\n=== Plotting Best Model Accuracy Curve ===")
        best_hist = hb.get_best_history()
        plot_hyperband_accuracy(best_hist)


if __name__ == "__main__":
    Main().run()