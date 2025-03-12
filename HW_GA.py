import numpy as np
import pandas as pd
import random
from collections import defaultdict

# Define the Play-Tennis dataset attributes
attributes = {
    'Outlook': ['Sunny', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Mild', 'Cool'],
    'Humidity': ['High', 'Normal'],
    'Wind': ['Weak', 'Strong']
}

# Sample Play-Tennis dataset from Chapter 3
def create_play_tennis_dataset():
    data = [
        ['Sunny', 'Hot', 'High', 'Weak', 'No'],
        ['Sunny', 'Hot', 'High', 'Strong', 'No'],
        ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
        ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
        ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
        ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
        ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
        ['Sunny', 'Mild', 'High', 'Weak', 'No'],
        ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
        ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
        ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
        ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
        ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
        ['Rain', 'Mild', 'High', 'Strong', 'No']
    ]
    columns = ['Outlook', 'Temperature', 'Humidity', 'Wind', 'PlayTennis']
    return pd.DataFrame(data, columns=columns)

# Convert the bit-string to a rule
def bitstring_to_rule(bitstring, attributes):
    """
    Convert a bit-string to a human-readable rule.
    Bit-string format:
    - For each attribute value, there's a bit indicating whether it's included in the rule (1) or not (0)
    - The sequence is [Outlook_Sunny, Outlook_Overcast, Outlook_Rain, Temp_Hot, Temp_Mild, ...]
    """
    rule_parts = []
    idx = 0
    
    for attr, values in attributes.items():
        attr_conditions = []
        for val in values:
            if bitstring[idx] == 1:
                attr_conditions.append(f"{attr}={val}")
            idx += 1
        
        if attr_conditions:
            if len(attr_conditions) == 1:
                rule_parts.append(attr_conditions[0])
            else:
                rule_parts.append(f"({' OR '.join(attr_conditions)})")
    
    if not rule_parts:
        return "True"  # Always matches
    return " AND ".join(rule_parts)

# Calculate the total length of the bit-string based on the attributes
def calculate_bitstring_length(attributes):
    return sum(len(values) for values in attributes.values())

# Initialize a random population of bit-strings
def initialize_population(pop_size, bitstring_length):
    return [np.random.randint(0, 2, bitstring_length).tolist() for _ in range(pop_size)]

# Evaluate the fitness of a bit-string
def evaluate_fitness(bitstring, dataset, attributes):
    # Special case: all zeros means no conditions (matches everything)
    if sum(bitstring) == 0:
        return 0.1  # Low fitness for trivial rules
    
    # Apply the rule to each instance in the dataset
    correct_predictions = 0
    total_instances = len(dataset)
    
    for _, instance in dataset.iterrows():
        if matches_rule(bitstring, instance, attributes):
            if instance['PlayTennis'] == 'Yes':
                correct_predictions += 1
        else:
            if instance['PlayTennis'] == 'No':
                correct_predictions += 1
    
    # Calculate accuracy
    accuracy = correct_predictions / total_instances
    
    # Reduce complexity penalty
    complexity_penalty = 0.001 * sum(bitstring) / len(bitstring)  # Reduced from 0.01 to 0.001
    
    return accuracy - complexity_penalty

# Check if an instance matches a rule
def matches_rule(bitstring, instance, attributes):
    idx = 0
    for attr, values in attributes.items():
        match_found = False
        for i, val in enumerate(values):
            if bitstring[idx + i] == 1 and instance[attr] == val:
                match_found = True
                break
        
        if not match_found and any(bitstring[idx:idx+len(values)]):
            # If no match is found for a condition that exists in the rule, the rule doesn't match
            return False
        
        idx += len(values)
    
    return True

# Select parents using tournament selection
def select_parents(population, fitnesses, tournament_size=3):
    parent1_idx = tournament_selection(fitnesses, tournament_size)
    parent2_idx = tournament_selection(fitnesses, tournament_size)
    return population[parent1_idx], population[parent2_idx]

# Tournament selection
def tournament_selection(fitnesses, tournament_size):
    candidates = random.sample(range(len(fitnesses)), tournament_size)
    best_idx = candidates[0]
    for idx in candidates:
        if fitnesses[idx] > fitnesses[best_idx]:
            best_idx = idx
    return best_idx

# Crossover operators
def single_point_crossover(parent1, parent2):
    """Single point crossover between two bit-strings."""
    if len(parent1) <= 1:
        return parent1.copy(), parent2.copy()
    
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def two_point_crossover(parent1, parent2):
    """Two point crossover between two bit-strings."""
    if len(parent1) <= 2:
        return parent1.copy(), parent2.copy()
    
    point1 = random.randint(1, len(parent1) - 2)
    point2 = random.randint(point1 + 1, len(parent1) - 1)
    
    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
    return child1, child2

def uniform_crossover(parent1, parent2, p=0.5):
    """Uniform crossover with probability p for each bit."""
    child1, child2 = [], []
    for i in range(len(parent1)):
        if random.random() < p:
            child1.append(parent1[i])
            child2.append(parent2[i])
        else:
            child1.append(parent2[i])
            child2.append(parent1[i])
    return child1, child2

def attribute_based_crossover(parent1, parent2, attributes):
    """
    Attribute-based crossover - maintains attribute boundaries.
    This keeps related bits together by crossing over at attribute boundaries.
    """
    child1, child2 = [], []
    idx = 0
    
    for attr, values in attributes.items():
        attr_length = len(values)
        
        # For each attribute, randomly decide which parent to take from
        if random.random() < 0.5:
            child1.extend(parent1[idx:idx+attr_length])
            child2.extend(parent2[idx:idx+attr_length])
        else:
            child1.extend(parent2[idx:idx+attr_length])
            child2.extend(parent1[idx:idx+attr_length])
        
        idx += attr_length
    
    return child1, child2

def validate_rule(rule_bitstring, attributes):
    # Check for conflicting conditions within attributes
    idx = 0
    for attr, values in attributes.items():
        active = sum(rule_bitstring[idx:idx+len(values)])
        if active > 1:
            return False  # Multiple values selected for single attribute
        idx += len(values)
    return True

# Modify mutation to prevent invalid rules
def mutate(bitstring, mutation_rate):
    result = bitstring.copy()
    for i in range(len(result)):
        if random.random() < mutation_rate:
            # Get attribute boundaries
            attr_idx = 0
            for attr, values in attributes.items():
                if i in range(attr_idx, attr_idx+len(values)):
                    # Flip entire attribute group
                    result[attr_idx:attr_idx+len(values)] = [0]*len(values)
                    result[i] = 1
                    break
                attr_idx += len(values)
    return result

# Add this function to evaluate the final rule more thoroughly
def evaluate_rule_performance(rule_bitstring, dataset, attributes):
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    
    for _, instance in dataset.iterrows():
        matches = matches_rule(rule_bitstring, instance, attributes)
        actual_positive = instance['PlayTennis'] == 'Yes'
        
        if matches and actual_positive:
            true_positives += 1
        elif matches and not actual_positive:
            false_positives += 1
        elif not matches and actual_positive:
            false_negatives += 1
        else:  # not matches and not actual_positive
            true_negatives += 1
    
    total = len(dataset)
    accuracy = (true_positives + true_negatives) / total
    
    # Handle division by zero
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    confusion_matrix = {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives
    }
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': confusion_matrix
    }

# Main genetic algorithm
def genetic_algorithm(dataset, attributes, pop_size=50, generations=100, replacement_rate=0.5, mutation_rate=0.01, crossover_type='single'):
    crossover_operators = {
        'single': single_point_crossover,
        'two_point': two_point_crossover,
        'uniform': uniform_crossover,
        'attribute': lambda p1, p2: attribute_based_crossover(p1, p2, attributes)
    }
    
    crossover_func = crossover_operators.get(crossover_type, single_point_crossover)
    
    bitstring_length = calculate_bitstring_length(attributes)
    population = initialize_population(pop_size, bitstring_length)
    
    # Keep track of the best solution found
    best_fitness = 0
    best_solution = None
    
    # History for tracking performance
    history = {
        'max_fitness': [],
        'avg_fitness': [],
        'best_rule': []
    }
    
    for generation in range(generations):
        # Evaluate fitness for each individual
        fitnesses = [evaluate_fitness(ind, dataset, attributes) for ind in population]
        
        # Track statistics
        max_fitness = max(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)
        history['max_fitness'].append(max_fitness)
        history['avg_fitness'].append(avg_fitness)
        
        # Find the best solution
        best_idx = fitnesses.index(max_fitness)
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_solution = population[best_idx]
            best_rule = bitstring_to_rule(best_solution, attributes)
            history['best_rule'].append((generation, best_rule, best_fitness))
        
        # Create the next generation
        num_to_replace = int(pop_size * replacement_rate)
        elites = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)[:pop_size-num_to_replace]
        
        next_population = [population[i] for i in elites]
        
        # Add new offspring
        while len(next_population) < pop_size:
            parent1, parent2 = select_parents(population, fitnesses)
            child1, child2 = crossover_func(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            
            next_population.append(child1)
            if len(next_population) < pop_size:
                next_population.append(child2)
        
        population = next_population
    
    # Return the best solution and its fitness
    best_rule = bitstring_to_rule(best_solution, attributes)
    print(f"Best Rule (Fitness: {best_fitness:.4f}): {best_rule}")
    
    return best_solution, best_fitness, history

# Run experiments with different parameters
def run_experiments(dataset, attributes):
    # Experiment configurations
    configurations = [
        # Balanced configuration
        {'pop_size': 50, 'replacement_rate': 0.3, 'mutation_rate': 0.05, 'crossover_type': 'attribute'},
        
        # High exploration configuration
        {'pop_size': 100, 'replacement_rate': 0.5, 'mutation_rate': 0.1, 'crossover_type': 'uniform'},
        
        # Conservative configuration
        {'pop_size': 30, 'replacement_rate': 0.2, 'mutation_rate': 0.02, 'crossover_type': 'attribute'},
        
        # Focused search configuration
        {'pop_size': 80, 'replacement_rate': 0.4, 'mutation_rate': 0.08, 'crossover_type': 'two_point'},
        
        # High diversity configuration
        {'pop_size': 150, 'replacement_rate': 0.6, 'mutation_rate': 0.15, 'crossover_type': 'uniform'},
        
        # Conservative with single-point crossover
        {'pop_size': 40, 'replacement_rate': 0.25, 'mutation_rate': 0.03, 'crossover_type': 'single'}
    ]
    
    results = []
    
    for i, config in enumerate(configurations):
        print(f"\nExperiment {i+1}: {config}")
        _, fitness, history = genetic_algorithm(
            dataset, 
            attributes, 
            pop_size=config['pop_size'],
            replacement_rate=config['replacement_rate'],
            mutation_rate=config['mutation_rate'],
            crossover_type=config['crossover_type']
        )
        
        results.append({
            'config': config,
            'fitness': fitness,
            'history': history
        })
    
    # Print summary of results
    print("\nExperiment Results Summary:")
    for i, result in enumerate(results):
        config = result['config']
        print(f"Experiment {i+1}: Pop={config['pop_size']}, Replace={config['replacement_rate']}, "
              f"Mutation={config['mutation_rate']}, Crossover={config['crossover_type']} -> "
              f"Best Fitness: {result['fitness']:.4f}")
    
    return results

# Main function
def main():
    # Create the Play-Tennis dataset
    dataset = create_play_tennis_dataset()
    print("Play-Tennis Dataset:")
    print(dataset)
    
    # Run the experiments
    results = run_experiments(dataset, attributes)
    
    # Find the best configuration
    best_result = max(results, key=lambda r: r['fitness'])
    best_config = best_result['config']
    
    print("\nBest Configuration:")
    print(f"Population Size: {best_config['pop_size']}")
    print(f"Replacement Rate: {best_config['replacement_rate']}")
    print(f"Mutation Rate: {best_config['mutation_rate']}")
    print(f"Crossover Type: {best_config['crossover_type']}")
    print(f"Best Fitness: {best_result['fitness']:.4f}")
    
    # Run a final, more thorough experiment with the best configuration
    print("\nRunning final experiment with best configuration...")
    _, fitness, history = genetic_algorithm(
        dataset, 
        attributes, 
        pop_size=best_config['pop_size'],
        replacement_rate=best_config['replacement_rate'],
        mutation_rate=best_config['mutation_rate'],
        crossover_type=best_config['crossover_type'],
        generations=200  # More generations for the final run
    )
    
    # Print the evolution of the best rule
    print("\nEvolution of the best rule:")
    for gen, rule, fit in history['best_rule']:
        print(f"Generation {gen}: {rule} (Fitness: {fit:.4f})")
    
    # In the main function, after finding the best configuration:
    print("\nDetailed evaluation of the best rule:")
    best_rule_bitstring, _, _ = genetic_algorithm(
        dataset, 
        attributes, 
        pop_size=best_config['pop_size'],
        replacement_rate=best_config['replacement_rate'],
        mutation_rate=best_config['mutation_rate'],
        crossover_type=best_config['crossover_type'],
        generations=200
    )

    # Get the human-readable rule
    best_rule_text = bitstring_to_rule(best_rule_bitstring, attributes)
    print(f"Best rule: {best_rule_text}")

    # Evaluate on the whole dataset
    metrics = evaluate_rule_performance(best_rule_bitstring, dataset, attributes)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print("Confusion Matrix:")
    print(f"  True Positives: {metrics['confusion_matrix']['true_positives']}")
    print(f"  False Positives: {metrics['confusion_matrix']['false_positives']}")
    print(f"  True Negatives: {metrics['confusion_matrix']['true_negatives']}")
    print(f"  False Negatives: {metrics['confusion_matrix']['false_negatives']}")

if __name__ == "__main__":
    main()