import matplotlib.pyplot as plt
from data import DataHandler
from random_search import RandomSearch


# Plot function

def plot_random_accuracy(history):
    train_acc = history["accuracy"]
    val_acc = history["val_accuracy"]

    plt.figure(figsize=(8, 6))
    plt.plot(train_acc, label="Train Accuracy", color="red")
    plt.plot(val_acc, label="Validation Accuracy", color="green")

    plt.title("Training vs Validation Accuracy (Best Random Search Model)")
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

        print("\n=== Running Random Search ===")
        rs = RandomSearch(data_handler=data, n_iter=10)
        best_cfg, best_score = rs.search()
        rs.save_results("random_search_results.json")

        print("\nBest Random Search Config:", best_cfg)
        print("Best Validation Accuracy:", best_score)

        print("\n=== Plotting Best Accuracy Curve ===")
        best_hist = rs.get_best_history()
        plot_random_accuracy(best_hist)


if __name__ == "__main__":
    Main().run()