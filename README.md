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

The project also includes a baseline CNN model and a reusable DataHandler class for dataset preprocessing.

src/
│
├── cnn.py
├── data.py
│
├── random_search.py
├── bayesian_search.py
├── hyperband_search.py
│
├── main_random_search.py
├── main_bayesian_search.py
├── main_hyperband_search.py
└── main_cnn.py

How to run Each Search Methods
- Baseline CNN (no hyperparameter search) - python src/main_cnn.py
- Random Search - python src/main_random_search.py 
  - Output includes the best hyperparameter, training and validation accuracy plot, and a json result
- Bayesian Optimization - python src/main_bayesian_search.py
  - Output includes the Gaussian Process-based search, best configuration, accuracy curve, and a json result
- Hyperband Search -  python src/main_hyperband_search.py
  - Output includes the hyperband bracket & rounds, best configuration, accuracy curve, and a json result

Each of the search method save a JSON file:
- random_search_results.json 
- bayesian_search_results.json 
- hyperband_results.json
And each of the JSON contains
- best_config 
- best_score (best validation accuracy)
- total_search_time 
- results (list of all tried configurations)

