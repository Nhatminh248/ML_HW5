import numpy as np
import pandas as pd
import random

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

attributes = {
    'Outlook':     ['Sunny', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Mild', 'Cool'],
    'Humidity':    ['High', 'Normal'],
    'Wind':        ['Weak', 'Strong'],
}


def create_play_tennis_dataset() -> pd.DataFrame:
    data = [
        ['Sunny',    'Hot',  'High',   'Weak',   'No'],
        ['Sunny',    'Hot',  'High',   'Strong', 'No'],
        ['Overcast', 'Hot',  'High',   'Weak',   'Yes'],
        ['Rain',     'Mild', 'High',   'Weak',   'Yes'],
        ['Rain',     'Cool', 'Normal', 'Weak',   'Yes'],
        ['Rain',     'Cool', 'Normal', 'Strong', 'No'],
        ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
        ['Sunny',    'Mild', 'High',   'Weak',   'No'],
        ['Sunny',    'Cool', 'Normal', 'Weak',   'Yes'],
        ['Rain',     'Mild', 'Normal', 'Weak',   'Yes'],
        ['Sunny',    'Mild', 'Normal', 'Strong', 'Yes'],
        ['Overcast', 'Mild', 'High',   'Strong', 'Yes'],
        ['Overcast', 'Hot',  'Normal', 'Weak',   'Yes'],
        ['Rain',     'Mild', 'High',   'Strong', 'No'],
    ]
    return pd.DataFrame(data, columns=['Outlook', 'Temperature', 'Humidity', 'Wind', 'PlayTennis'])


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def calculate_bitstring_length(attributes: dict) -> int:
    return sum(len(v) for v in attributes.values())


def bitstring_to_rule(bitstring: list, attributes: dict) -> str:
    """Convert a bit-string to a human-readable conjunctive rule."""
    rule_parts = []
    idx = 0
    for attr, values in attributes.items():
        active = [values[i] for i in range(len(values)) if bitstring[idx + i] == 1]
        if len(active) == 1:
            rule_parts.append(f"{attr}={active[0]}")
        elif len(active) > 1:
            rule_parts.append(f"({' OR '.join(f'{attr}={v}' for v in active)})")
        idx += len(values)
    return " AND ".join(rule_parts) if rule_parts else "True (no conditions)"


def matches_rule(bitstring: list, instance: pd.Series, attributes: dict) -> bool:
    """
    Return True if `instance` satisfies every condition in the rule.
    An attribute whose bits are all 0 is treated as a wildcard (no constraint).
    """
    idx = 0
    for attr, values in attributes.items():
        attr_bits = bitstring[idx: idx + len(values)]
        # Wildcard — no constraint on this attribute
        if not any(attr_bits):
            idx += len(values)
            continue
        # At least one bit set — instance must match one of the active values
        matched = any(attr_bits[i] == 1 and instance[attr] == values[i]
                      for i in range(len(values)))
        if not matched:
            return False
        idx += len(values)
    return True


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------

def evaluate_fitness(bitstring: list, dataset: pd.DataFrame, attributes: dict) -> float:
    """
    Laplace accuracy over instances that match the rule.

        fitness = (TP + 1) / (TP + FP + 2) - complexity_penalty

    This rewards precision on the covered set rather than overall accuracy,
    preventing the GA from collapsing to trivially broad rules.
    A rule with no active bits is given fitness 0.
    """
    if sum(bitstring) == 0:
        return 0.0

    tp = fp = 0
    for _, instance in dataset.iterrows():
        if matches_rule(bitstring, instance, attributes):
            if instance['PlayTennis'] == 'Yes':
                tp += 1
            else:
                fp += 1

    laplace = (tp + 1) / (tp + fp + 2)
    complexity_penalty = 0.001 * sum(bitstring) / len(bitstring)
    return laplace - complexity_penalty


# ---------------------------------------------------------------------------
# Evaluation (reporting only — not used by GA internally)
# ---------------------------------------------------------------------------

def evaluate_rule_performance(bitstring: list, dataset: pd.DataFrame, attributes: dict) -> dict:
    tp = fp = tn = fn = 0
    for _, instance in dataset.iterrows():
        matched = matches_rule(bitstring, instance, attributes)
        positive = instance['PlayTennis'] == 'Yes'
        if matched and positive:
            tp += 1
        elif matched and not positive:
            fp += 1
        elif not matched and positive:
            fn += 1
        else:
            tn += 1

    total = len(dataset)
    accuracy  = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    laplace   = (tp + 1) / (tp + fp + 2)

    return {
        'accuracy':  accuracy,
        'precision': precision,
        'recall':    recall,
        'f1_score':  f1,
        'laplace':   laplace,
        'confusion_matrix': {
            'true_positives':  tp,
            'false_positives': fp,
            'true_negatives':  tn,
            'false_negatives': fn,
        },
    }


# ---------------------------------------------------------------------------
# Genetic operators
# ---------------------------------------------------------------------------

def initialize_population(pop_size: int, bitstring_length: int) -> list:
    return [np.random.randint(0, 2, bitstring_length).tolist()
            for _ in range(pop_size)]


def tournament_selection(fitnesses: list, tournament_size: int = 3) -> int:
    candidates = random.sample(range(len(fitnesses)), tournament_size)
    return max(candidates, key=lambda i: fitnesses[i])


def select_parents(population: list, fitnesses: list, tournament_size: int = 3):
    i = tournament_selection(fitnesses, tournament_size)
    j = tournament_selection(fitnesses, tournament_size)
    return population[i], population[j]


def single_point_crossover(p1, p2):
    if len(p1) <= 1:
        return p1.copy(), p2.copy()
    pt = random.randint(1, len(p1) - 1)
    return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]


def two_point_crossover(p1, p2):
    if len(p1) <= 2:
        return p1.copy(), p2.copy()
    a = random.randint(1, len(p1) - 2)
    b = random.randint(a + 1, len(p1) - 1)
    return (p1[:a] + p2[a:b] + p1[b:],
            p2[:a] + p1[a:b] + p2[b:])


def uniform_crossover(p1, p2, prob=0.5):
    c1, c2 = [], []
    for b1, b2 in zip(p1, p2):
        if random.random() < prob:
            c1.append(b1); c2.append(b2)
        else:
            c1.append(b2); c2.append(b1)
    return c1, c2


def attribute_based_crossover(p1, p2, attributes):
    """Crossover at attribute boundaries to keep attribute segments intact."""
    c1, c2 = [], []
    idx = 0
    for values in attributes.values():
        n = len(values)
        if random.random() < 0.5:
            c1.extend(p1[idx: idx + n])
            c2.extend(p2[idx: idx + n])
        else:
            c1.extend(p2[idx: idx + n])
            c2.extend(p1[idx: idx + n])
        idx += n
    return c1, c2


def mutate(bitstring: list, mutation_rate: float, attributes: dict) -> list:
    """
    Flip a bit with probability `mutation_rate`.
    When a bit is flipped on, the rest of that attribute's bits are cleared
    to keep the encoding valid (at most one active value per attribute after
    a mutation event; crossover may produce multi-value conditions, which is
    intentional and represents OR-conditions within an attribute).
    """
    result = bitstring.copy()
    idx = 0
    for values in attributes.values():
        n = len(values)
        for i in range(n):
            if random.random() < mutation_rate:
                # Clear all bits for this attribute, then set the flipped one
                result[idx: idx + n] = [0] * n
                result[idx + i] = 1
                break   # one mutation event per attribute per call
        idx += n
    return result


# ---------------------------------------------------------------------------
# Genetic algorithm
# ---------------------------------------------------------------------------

CROSSOVER_FUNCS = {
    'single':     single_point_crossover,
    'two_point':  two_point_crossover,
    'uniform':    uniform_crossover,
    'attribute':  None,   # handled specially (needs attributes arg)
}


def genetic_algorithm(
    dataset: pd.DataFrame,
    attributes: dict,
    pop_size: int = 50,
    generations: int = 100,
    replacement_rate: float = 0.5,
    mutation_rate: float = 0.01,
    crossover_type: str = 'attribute',
) -> tuple:
    length = calculate_bitstring_length(attributes)
    population = initialize_population(pop_size, length)

    best_fitness = -1.0
    best_solution = population[0]
    history = {'max_fitness': [], 'avg_fitness': [], 'best_rule': []}

    def crossover(p1, p2):
        if crossover_type == 'attribute':
            return attribute_based_crossover(p1, p2, attributes)
        return CROSSOVER_FUNCS[crossover_type](p1, p2)

    for gen in range(generations):
        fitnesses = [evaluate_fitness(ind, dataset, attributes) for ind in population]

        max_f = max(fitnesses)
        avg_f = sum(fitnesses) / len(fitnesses)
        history['max_fitness'].append(max_f)
        history['avg_fitness'].append(avg_f)

        best_idx = fitnesses.index(max_f)
        if max_f > best_fitness:
            best_fitness = max_f
            best_solution = population[best_idx]
            history['best_rule'].append(
                (gen, bitstring_to_rule(best_solution, attributes), best_fitness)
            )

        # Elitism: keep top (1 - replacement_rate) fraction unchanged
        n_replace = int(pop_size * replacement_rate)
        elite_idxs = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
        next_pop = [population[i] for i in elite_idxs[: pop_size - n_replace]]

        while len(next_pop) < pop_size:
            p1, p2 = select_parents(population, fitnesses)
            c1, c2 = crossover(p1, p2)
            next_pop.append(mutate(c1, mutation_rate, attributes))
            if len(next_pop) < pop_size:
                next_pop.append(mutate(c2, mutation_rate, attributes))

        population = next_pop

    print(f"  Best Rule (fitness={best_fitness:.4f}): {bitstring_to_rule(best_solution, attributes)}")
    return best_solution, best_fitness, history


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

CONFIGURATIONS = [
    {'pop_size': 50,  'replacement_rate': 0.3, 'mutation_rate': 0.05, 'crossover_type': 'attribute'},
    {'pop_size': 100, 'replacement_rate': 0.5, 'mutation_rate': 0.10, 'crossover_type': 'uniform'},
    {'pop_size': 30,  'replacement_rate': 0.2, 'mutation_rate': 0.02, 'crossover_type': 'attribute'},
    {'pop_size': 80,  'replacement_rate': 0.4, 'mutation_rate': 0.08, 'crossover_type': 'two_point'},
    {'pop_size': 150, 'replacement_rate': 0.6, 'mutation_rate': 0.15, 'crossover_type': 'uniform'},
    {'pop_size': 40,  'replacement_rate': 0.25,'mutation_rate': 0.03, 'crossover_type': 'single'},
]


def run_experiments(dataset: pd.DataFrame, attributes: dict) -> list:
    results = []
    for i, cfg in enumerate(CONFIGURATIONS):
        print(f"\nExperiment {i + 1}: {cfg}")
        _, fitness, history = genetic_algorithm(dataset, attributes, **cfg)
        results.append({'config': cfg, 'fitness': fitness, 'history': history})

    print("\n--- Experiment Results Summary ---")
    for i, r in enumerate(results):
        c = r['config']
        print(f"  Exp {i+1}: pop={c['pop_size']} replace={c['replacement_rate']} "
              f"mut={c['mutation_rate']} xover={c['crossover_type']} "
              f"-> fitness={r['fitness']:.4f}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    dataset = create_play_tennis_dataset()
    print("=== Play-Tennis Dataset (14 instances) ===")
    print(dataset.to_string(index=False))

    print("\n=== Running Experiments ===")
    results = run_experiments(dataset, attributes)

    best_result = max(results, key=lambda r: r['fitness'])
    best_cfg    = best_result['config']

    print("\n=== Best Configuration ===")
    for k, v in best_cfg.items():
        print(f"  {k}: {v}")
    print(f"  Fitness: {best_result['fitness']:.4f}")

    print("\n=== Final Run with Best Configuration (200 generations) ===")
    best_bitstring, best_fitness, history = genetic_algorithm(
        dataset, attributes, generations=200, **best_cfg
    )

    print("\n--- Rule Evolution ---")
    for gen, rule, fit in history['best_rule']:
        print(f"  Gen {gen:>3}: {rule}  (fitness={fit:.4f})")

    print("\n--- Detailed Evaluation of Best Rule ---")
    rule_text = bitstring_to_rule(best_bitstring, attributes)
    print(f"  Rule     : {rule_text}")
    m = evaluate_rule_performance(best_bitstring, dataset, attributes)
    print(f"  Laplace  : {m['laplace']:.4f}")
    print(f"  Accuracy : {m['accuracy']:.4f}")
    print(f"  Precision: {m['precision']:.4f}")
    print(f"  Recall   : {m['recall']:.4f}")
    print(f"  F1 Score : {m['f1_score']:.4f}")
    cm = m['confusion_matrix']
    print(f"  Confusion: TP={cm['true_positives']}  FP={cm['false_positives']}  "
          f"TN={cm['true_negatives']}  FN={cm['false_negatives']}")


if __name__ == "__main__":
    main()
