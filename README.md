CIFAR-10 HYPERPARAMETER OPTIMIZATION PROJECT

This project implements and compare three hyperparameter optimization techniques on a Convolutional Neural Network(CNN)
trained on the CIFAR-10 image classification dataset.
- Random Search
- Bayesian Optimization
- Hyperband (Successive Halving)

Each methods tune the same parameters
- Learning Rate
- Batch Size
- Number of Epochs

The project also includes a baseline CNN model and a reusable DataHandler class for dataset preprocessing, as well as JSON result
file for each search method.

**PROJECT STRUCTURE**

src/
│
├── cnn.py                        # CNN architecture + training utilities
├── data.py                       # DataHandler: load & preprocess CIFAR-10
│
├── random_search.py              # Random Search Implementation
├── bayesian_search.py            # Bayesian Optimization (Gaussian Process)
├── hyperband_search.py           # Hyperband (Successive Halving)
│
├── main_random_search.py         # Run Random Search
├── main_bayesian_search.py       # Run Bayesian Search
├── main_hyperband_search.py      # Run Hyperband Search
├── main_cnn.py                   # Run baseline cnn
│
├── random_search_results.json    # (Generated after running Random Search)
├── bayesian_search_results.json  # (Generated after running Bayesian Search)
└── hyperband_results.json        # (Generated after running Hyperband)

**ENVIRONMENT SETUP**
- Clone the repository or download the project folder
- install dependencies: pip install tensorflow numpy matplotlib scikit-learn scipy


**HOW TO RUN EACH SEARCH METHOD**

ALL COMMANDS ASSUME YOU ARE IN THE PROJECT ROOT.

- **Baseline CNN (no hyperparameter search)** - python src/main_cnn.py
Outputs:
- Model Summary
- Training and validation accuracy curve
- Test accuracy
- Sample prediction visualization

- **RANDOM SEARCH** - python src/main_random_search.py
This script:
- Runs N random trials
- Trains a CNN with each random configuration
- Logs:
  - Best hyperparameters
  - Validation accuracy for each trial
  - Training curve for best model
- Saves results to - random_search_results.json

- **BAYESIAN OPTIMIZATION** - python src/main_bayesian_search.py
This script:
- Uses a Gaussian Process (RBF kernel)
- Applies Expected Improvement (EI) acquisition
- Selects the next hyperparameter set based on EI
- Logs:
  - Initial random samples
  - Bayesian selected samples
  - Best hyperparameter and accuracy curve
- Saves results to - bayesian_search_results.json

- **HYPERBAND SEARCH** -  python src/main_hyperband_search.py
 This script:
- Creates multiple brackets based on Hyperband theory
- Performs Successive Halving within each bracket
- Adaptively allocate training epochs
- Logs:
  - Each round's survivor
  - Best configuration found
  - Learning curve of the best model
- Saves results to - hyperband_search_results.json

**JSON FILE CONTENTS**
Each search method generates a JSON containing:
{
  "best_config": { ... },
  "best_score": <best_validation_accuracy>,
  "total_search_time": <seconds>,
  "results": [
     {
       "config": {...},
       "val_acc": ...,
       "train_time": ...,
       "history": {
           "accuracy": [...],
           "loss": [...],
           "val_accuracy": [...],
           "val_loss": [...]
       }
     },
     ...
  ]
}

**NOTE**
- All hyperparameter searches uses the same CNN architecture.
- All searches use the same search space to ensure fairness:
  - LR ∈ [0.0001, 0.01]
  - Batch size ∈ {8,16,32,64,128}
  - Epochs ∈ {5,10,…,50}
- Early stopping with patience = 3 is applied in every model.
- The project supports deterministic runs when manually enabling seeds.


**CONTRIBUTORS**
- Godsfavour Ogbonna
- Sylvester Mensah
- Daniel Griffith
- Albert Asamoah

