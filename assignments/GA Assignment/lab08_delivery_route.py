"""
Genetic Algorithm for Delivery Route Optimisation
-------------------------------------------------
Depot = A. Delivery locations = B, C, D, E, F, G.
Goal: find the shortest round trip A -> (B..G in some order) -> A.

Chromosome  : a permutation of the six delivery locations, e.g. ['C','F','E','B','D','G'].
              The depot A is added at the start and end only when the distance is
              calculated, so A is never touched by crossover or mutation.
Fitness     : total round-trip distance in km (LOWER is better).
Selection   : tournament selection (size 3).
Crossover   : Order Crossover (OX) - always gives a valid permutation.
Mutation    : swap two randomly chosen locations.
Elitism     : the best routes are copied unchanged into the next generation.
"""

import random
import itertools

# ------------------------------------------------------------------
# 1. Dataset: symmetric distance matrix (km) from the assignment
# ------------------------------------------------------------------
LOCATIONS = ["A", "B", "C", "D", "E", "F", "G"]
DISTANCE = {
    "A": {"A": 0, "B": 4, "C": 8, "D": 6, "E": 7, "F": 9, "G": 5},
    "B": {"A": 4, "B": 0, "C": 5, "D": 7, "E": 3, "F": 8, "G": 6},
    "C": {"A": 8, "B": 5, "C": 0, "D": 4, "E": 6, "F": 3, "G": 7},
    "D": {"A": 6, "B": 7, "C": 4, "D": 0, "E": 5, "F": 6, "G": 8},
    "E": {"A": 7, "B": 3, "C": 6, "D": 5, "E": 0, "F": 4, "G": 6},
    "F": {"A": 9, "B": 8, "C": 3, "D": 6, "E": 4, "F": 0, "G": 5},
    "G": {"A": 5, "B": 6, "C": 7, "D": 8, "E": 6, "F": 5, "G": 0},
}
DEPOT = "A"
DELIVERY_STOPS = ["B", "C", "D", "E", "F", "G"]

# ------------------------------------------------------------------
# 2. GA settings
# ------------------------------------------------------------------
RANDOM_SEED = 42        # fixed seed -> the experiment is reproducible
POPULATION_SIZE = 50
GENERATIONS = 100       # assignment requires at least 50
TOURNAMENT_SIZE = 3
MUTATION_RATE = 0.40    # chance that a child is mutated (high enough to keep diversity)
ELITE_COUNT = 2         # best routes copied unchanged


# ------------------------------------------------------------------
# 3. Fitness: total distance including the return journey to depot A
# ------------------------------------------------------------------
def total_distance(chromosome):
    full_route = [DEPOT] + list(chromosome) + [DEPOT]
    return sum(DISTANCE[full_route[i]][full_route[i + 1]]
               for i in range(len(full_route) - 1))


# ------------------------------------------------------------------
# 4. Initial population of valid routes (random shuffles of B-G)
# ------------------------------------------------------------------
def create_initial_population(size):
    population = []
    for _ in range(size):
        route = DELIVERY_STOPS[:]
        random.shuffle(route)
        population.append(route)
    return population


# ------------------------------------------------------------------
# 5. Parent selection: tournament of three (lowest distance wins)
# ------------------------------------------------------------------
def tournament_selection(population):
    contestants = random.sample(population, TOURNAMENT_SIZE)
    return min(contestants, key=total_distance)


# ------------------------------------------------------------------
# 6. Route-preserving crossover: Order Crossover (OX)
#    Copy a slice from parent 1, then fill the remaining positions with
#    the missing locations in the order they appear in parent 2.
# ------------------------------------------------------------------
def order_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end + 1] = parent1[start:end + 1]          # slice from parent 1
    remaining = [stop for stop in parent2 if stop not in child]
    fill_positions = [i for i in range(size) if child[i] is None]
    for position, stop in zip(fill_positions, remaining):  # rest from parent 2
        child[position] = stop
    return child


# ------------------------------------------------------------------
# 7. Mutation: swap two locations (the route stays valid)
# ------------------------------------------------------------------
def swap_mutation(route):
    mutated = route[:]
    i, j = random.sample(range(len(mutated)), 2)
    mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated


# ------------------------------------------------------------------
# 8. Main GA loop
# ------------------------------------------------------------------
def run_genetic_algorithm():
    random.seed(RANDOM_SEED)
    population = create_initial_population(POPULATION_SIZE)
    best_route = min(population, key=total_distance)
    print(f"Generation   0: best distance = {total_distance(best_route)} km")

    for generation in range(1, GENERATIONS + 1):
        # sort so the shortest routes come first
        population.sort(key=total_distance)

        # Elitism: keep the best routes unchanged
        next_population = [route[:] for route in population[:ELITE_COUNT]]

        # Fill the rest with children of selected parents
        while len(next_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child = order_crossover(parent1, parent2)
            if random.random() < MUTATION_RATE:
                child = swap_mutation(child)
            next_population.append(child)

        population = next_population
        generation_best = min(population, key=total_distance)
        if total_distance(generation_best) < total_distance(best_route):
            best_route = generation_best[:]
        if generation % 10 == 0:
            print(f"Generation {generation:3d}: best distance = "
                  f"{total_distance(best_route)} km")

    return best_route


# ------------------------------------------------------------------
# 9. Run and report
# ------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Random seed = {RANDOM_SEED}, population = {POPULATION_SIZE}, "
          f"generations = {GENERATIONS}\n")
    best = run_genetic_algorithm()
    best_full_route = [DEPOT] + best + [DEPOT]
    print("\n=== RESULT ===")
    print("Best route     :", " -> ".join(best_full_route))
    print("Total distance :", total_distance(best), "km")

    # Optional check: with only 6! = 720 possible routes, brute force is easy,
    # so we can confirm whether the GA found the true optimum.
    optimum = min(itertools.permutations(DELIVERY_STOPS), key=total_distance)
    print("\nBrute-force check (720 routes):",
          " -> ".join([DEPOT] + list(optimum) + [DEPOT]),
          "=", total_distance(optimum), "km")