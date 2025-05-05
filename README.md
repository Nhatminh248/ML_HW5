# Genetic Algorithm for Rule Discovery in Play-Tennis Dataset

This program uses a genetic algorithm to discover rules for predicting whether to play tennis based on the weather conditions (Outlook, Temperature, Humidity, Wind). It works with the Play-Tennis dataset and applies evolutionary algorithms (selection, crossover, mutation) to evolve bit-string rules that can predict the outcome.

## Requirements

- Python 3.x
- Libraries:
  - `numpy`
  - `pandas`

You can install the required libraries using `pip`:

```bash
pip install numpy pandas
```

## Program Overview
1.	Play-Tennis Dataset: A simple dataset containing weather conditions (Outlook, Temperature, Humidity, Wind) and the decision to play tennis (PlayTennis).
2.	Genetic Algorithm:
- Population Initialization: Random bit-strings are initialized as potential solutions.
- Fitness Evaluation: Each bit-string (rule) is evaluated based on its accuracy in predicting the outcome of the dataset.
- Selection: Tournament selection is used to pick parents for crossover.
- Crossover: Various crossover methods (single-point, two-point, uniform, attribute-based) are available for combining parent bit-strings.
- Mutation: Bit-strings undergo mutation with a specified mutation rate.
- Rule Conversion: Bit-strings are converted to human-readable rules to understand the evolved solution.

## Functions
- create_play_tennis_dataset(): Generates the Play-Tennis dataset.
- bitstring_to_rule(): Converts a bit-string to a human-readable rule.
- initialize_population(): Initializes the population of bit-strings.
- evaluate_fitness(): Evaluates the fitness of a bit-string by checking its prediction accuracy.
- select_parents(): Selects two parents using tournament selection.
- crossover_operators: Contains functions for different crossover techniques.
- mutate(): Mutates a bit-string with a given mutation rate.
- genetic_algorithm(): Main function to run the genetic algorithm.
- run_experiments(): Runs different experiment configurations to find the best setup.
- evaluate_rule_performance(): Evaluates the performance of a generated rule on the dataset.

## How to Run
Run the program directly with the following command:

```bash
python HW_GA.py
```

This will run the genetic algorithm to evolve rules for predicting the “PlayTennis” outcome based on the weather conditions and display the results.

## Results
The program prints the following:
- The best rule found, along with its fitness score.
- A summary of the evolutionary process with fitness and rule history.
- Detailed evaluation metrics such as accuracy, precision, recall, F1 score, and confusion matrix for the best rule.


## [Updated] Synthetic Data Generation

Enhances the GA's performance by expanding the training dataset while preserving original patterns.

### Features:
- **Pattern Preservation**: Maintains original attribute-outcome relationships
- **Customizable**: Control dataset size (`num_samples`) and randomness (`noise_level`)
- **Auto-balancing**: Ensures equal class distribution

### Technical Approach:
- First-pass analysis of attribute-value-outcome probabilities
- Bayesian-style probability updates when generating each sample
- Smart handling of rare attribute combinations
- Optional class rebalancing
