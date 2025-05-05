import random
import pandas as pd
from collections import defaultdict

def analyze_dataset_patterns(dataset):
    """Analyze the original dataset to extract probability distributions."""
    patterns = defaultdict(int)  # Changed from nested defaultdict
    total = len(dataset)
    
    for _, row in dataset.iterrows():
        outcome = row['PlayTennis']
        for attr in ['Outlook', 'Temperature', 'Humidity', 'Wind']:
            key = (attr, row[attr], outcome)
            patterns[key] += 1
    
    # Convert counts to probabilities
    prob_table = defaultdict(dict)
    for (attr, val, outcome), count in patterns.items():
        # Calculate total counts for this (attribute, value) pair
        total_attr_val = sum(v for k, v in patterns.items() 
                          if k[0] == attr and k[1] == val)
        
        if total_attr_val > 0:
            prob_table[(attr, val)][outcome] = count / total_attr_val
    
    return prob_table

def generate_synthetic_data(original_data, num_samples=1000, noise_level=0.05):
    """
    Generate synthetic Play-Tennis data while preserving original patterns.
    """
    prob_table = analyze_dataset_patterns(original_data)
    attributes = {
        'Outlook': ['Sunny', 'Overcast', 'Rain'],
        'Temperature': ['Hot', 'Mild', 'Cool'],
        'Humidity': ['High', 'Normal'],
        'Wind': ['Weak', 'Strong']
    }
    
    synthetic_data = []
    
    for _ in range(num_samples):
        instance = {}
        outcome_probs = {'Yes': 0.5, 'No': 0.5}  # Start with equal priors
        
        # Generate attributes based on original distributions
        for attr, values in attributes.items():
            # Get distribution from original data
            value_counts = original_data[attr].value_counts(normalize=True)
            
            # Sample attribute value
            if random.random() < noise_level:
                # Introduce some randomness
                val = random.choice(values)
            else:
                # Follow original distribution
                val = random.choices(
                    list(value_counts.index),
                    weights=list(value_counts.values),
                    k=1
                )[0]
            
            instance[attr] = val
            
            # Update outcome probabilities based on this attribute
            if (attr, val) in prob_table:
                for outcome, prob in prob_table[(attr, val)].items():
                    outcome_probs[outcome] *= prob
        
        # Determine outcome
        if random.random() < noise_level:
            # Introduce some noise in outcomes
            play_tennis = random.choice(['Yes', 'No'])
        else:
            # Normalize probabilities and choose outcome
            total = sum(outcome_probs.values())
            if total == 0:
                play_tennis = random.choice(['Yes', 'No'])
            else:
                play_tennis = random.choices(
                    ['Yes', 'No'],
                    weights=[outcome_probs['Yes']/total, outcome_probs['No']/total],
                    k=1
                )[0]
        
        instance['PlayTennis'] = play_tennis
        synthetic_data.append(instance)
    
    return pd.DataFrame(synthetic_data)

def create_balanced_dataset(original_data, synthetic_data):
    """
    Combine original and synthetic data, ensuring class balance.
    """
    combined = pd.concat([original_data, synthetic_data])
    
    # Balance the classes if needed
    yes_count = combined[combined['PlayTennis'] == 'Yes'].shape[0]
    no_count = combined[combined['PlayTennis'] == 'No'].shape[0]
    
    if yes_count > no_count:
        # Sample more 'No' instances from synthetic data
        extra_no = synthetic_data[synthetic_data['PlayTennis'] == 'No'].sample(
            yes_count - no_count, replace=True)
        combined = pd.concat([combined, extra_no])
    elif no_count > yes_count:
        # Sample more 'Yes' instances from synthetic data
        extra_yes = synthetic_data[synthetic_data['PlayTennis'] == 'Yes'].sample(
            no_count - yes_count, replace=True)
        combined = pd.concat([combined, extra_yes])
    
    return combined.sample(frac=1).reset_index(drop=True)  # Shuffle the data
