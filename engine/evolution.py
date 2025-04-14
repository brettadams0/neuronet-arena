# engine/evolution.py
# 🧬 Evolution strategy: selects top brains and mutates them to create a new generation

import random

def evolve_population(brains, fitnesses, retain=0.4, mutate_rate=0.2):
    # Sort by fitness descending
    scored = sorted(zip(fitnesses, brains), key=lambda x: x[0], reverse=True)
    survivors = [brain.clone() for _, brain in scored[:max(1, int(len(brains) * retain))]]

    # Fill the population
    new_population = []
    while len(new_population) < len(brains):
        parent = random.choice(survivors).clone()
        parent.mutate(rate=mutate_rate)
        new_population.append(parent)

    return new_population[:len(brains)]
