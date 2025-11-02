import Reporter
import numpy as np
from numba import jit, njit, prange


@njit(cache=True)
def _tour_length(route: np.ndarray, dist: np.ndarray) -> float:
    # Compute total tour length including return to start
    total = 0.0
    n = len(route)
    for i in range(n):
        total += dist[route[i], route[(i + 1) % n]]
    return total


@njit(cache=True)
def _is_valid_path(path: np.ndarray, distanceMatrix: np.ndarray) -> bool:
    n = len(path)
    for i in range(n):
        if not np.isfinite(distanceMatrix[path[i], path[(i + 1) % n]]):
            return False
    return True


@njit(cache=True)
def _evaluate_population(pop: np.ndarray, dist: np.ndarray) -> np.ndarray:
    n = pop.shape[0]
    fitness = np.empty(n, dtype=np.float64)
    for i in prange(n):
        fitness[i] = _tour_length(pop[i], dist)
    return fitness


@njit
def _select_parent_pairs(
    pop: np.ndarray,
    fitness: np.ndarray,
    tournament_k: int,
    n_pairs: int,
    rng_state: np.ndarray,
) -> np.ndarray:
    """Pre-select all parent pairs at once for better cache efficiency"""
    pairs = np.empty(
        (n_pairs, 2), dtype=np.int32
    )  # Store indices instead of full solutions
    pop_size = len(pop)

    for i in range(n_pairs):
        # Select two parents using tournament selection
        parent1_idx = _tournament_select_index(
            pop_size, fitness, tournament_k, rng_state
        )
        parent2_idx = _tournament_select_index(
            pop_size, fitness, tournament_k, rng_state
        )
        while parent2_idx == parent1_idx:  # Ensure different parents
            parent2_idx = _tournament_select_index(
                pop_size, fitness, tournament_k, rng_state
            )
        pairs[i, 0] = parent1_idx
        pairs[i, 1] = parent2_idx
    return pairs


@njit
def _tournament_select_index(
    pop_size: int, fitness: np.ndarray, k: int, rng_state: np.ndarray
) -> int:
    """Tournament selection returning index instead of copying solution"""
    best_idx = int(np.random.random() * pop_size)
    best_fitness = fitness[best_idx]

    for _ in range(k - 1):
        idx = int(np.random.random() * pop_size)
        if fitness[idx] < best_fitness:
            best_idx = idx
            best_fitness = fitness[idx]
    return best_idx


@njit(cache=True)
def _scramble_mutation(
    solution: np.ndarray, distanceMatrix: np.ndarray, rng_state: np.ndarray
) -> np.ndarray:
    n = solution.size
    if n <= 1:
        return solution.copy()

    attempts = 20
    orig = solution.copy()

    for _ in range(attempts):
        # Generate random indices
        i = int(np.random.random() * n)
        j = int(np.random.random() * n)
        if i > j:
            i, j = j, i
        if i == j:
            continue

        cand = orig.copy()
        sub = cand[i : j + 1].copy()
        # Simple Fisher-Yates shuffle
        for idx in range(len(sub) - 1, 0, -1):
            p = int(np.random.random() * (idx + 1))
            sub[idx], sub[p] = sub[p], sub[idx]
        cand[i : j + 1] = sub

        if _is_valid_path(cand, distanceMatrix):
            return cand
    return orig


@njit(cache=True)
def _inversion_mutation(
    solution: np.ndarray, distanceMatrix: np.ndarray, rng_state: np.ndarray
) -> np.ndarray:
    n = solution.size
    if n <= 1:
        return solution.copy()

    attempts = 20
    orig = solution.copy()

    for _ in range(attempts):
        # Generate random indices
        i = int(np.random.random() * n)
        j = int(np.random.random() * n)
        if i > j:
            i, j = j, i
        if i == j:
            continue

        cand = orig.copy()
        # Reverse the subsequence
        cand[i : j + 1] = cand[i : j + 1][::-1]

        if _is_valid_path(cand, distanceMatrix):
            return cand

    return orig


@njit(cache=True)
def _tournament_select_one(
    population: np.ndarray, fitness: np.ndarray, k: int, rng_state: np.ndarray
) -> np.ndarray:
    pop_size = len(population)
    best_idx = int(np.random.random() * pop_size)
    best_fitness = fitness[best_idx]

    for _ in range(k - 1):
        idx = int(np.random.random() * pop_size)
        if fitness[idx] < best_fitness:
            best_idx = idx
            best_fitness = fitness[idx]

    return population[best_idx].copy()


@njit(cache=True)
def _edge_recombination(
    parent1: np.ndarray, parent2: np.ndarray, rng_state: np.ndarray
) -> np.ndarray:
    n = len(parent1)
    # Use fixed-size arrays for better performance
    edge_map = np.zeros((n, 4), dtype=np.int32)  # Maximum 4 edges per city
    edge_counts = np.zeros(n, dtype=np.int32)

    # Pre-compute parent indices for faster lookup
    p1_indices = np.zeros(n, dtype=np.int32)
    p2_indices = np.zeros(n, dtype=np.int32)
    for i in range(n):
        p1_indices[parent1[i]] = i
        p2_indices[parent2[i]] = i

    # Build edge map more efficiently
    for i in range(n):
        curr = parent1[i]
        # Add edges from parent1
        prev = parent1[(i - 1) % n]
        next = parent1[(i + 1) % n]

        if edge_counts[curr] < 4:
            edge_map[curr, edge_counts[curr]] = prev
            edge_counts[curr] += 1
        if edge_counts[curr] < 4:
            edge_map[curr, edge_counts[curr]] = next
            edge_counts[curr] += 1

        # Add unique edges from parent2
        curr = parent2[i]
        prev = parent2[(i - 1) % n]
        next = parent2[(i + 1) % n]

        # Only add if not already present and space available
        if edge_counts[curr] < 4 and not _contains(
            edge_map[curr, : edge_counts[curr]], prev
        ):
            edge_map[curr, edge_counts[curr]] = prev
            edge_counts[curr] += 1
        if edge_counts[curr] < 4 and not _contains(
            edge_map[curr, : edge_counts[curr]], next
        ):
            edge_map[curr, edge_counts[curr]] = next
            edge_counts[curr] += 1

    return _construct_offspring(edge_map, edge_counts, n, rng_state)


@njit(cache=True)
def _contains(arr: np.ndarray, val: int) -> bool:
    """Fast check for value in small array"""
    for x in arr:
        if x == val:
            return True
    return False


@njit(cache=True)
def _construct_offspring(
    edge_map: np.ndarray, edge_counts: np.ndarray, n: int, rng_state: np.ndarray
) -> np.ndarray:
    """Separate offspring construction for better optimization"""
    offspring = np.empty(n, dtype=np.int32)
    used = np.zeros(n, dtype=np.bool_)

    # Start with random city
    current = int(np.random.random() * n)

    for pos in range(n):
        offspring[pos] = current
        used[current] = True

        if pos < n - 1:
            # Find next city more efficiently
            neighbors = edge_map[current, : edge_counts[current]]
            if len(neighbors) > 0:
                min_edges = n + 1
                next_city = -1

                for neighbor in neighbors:
                    if not used[neighbor]:
                        count = edge_counts[neighbor]
                        if count < min_edges:
                            min_edges = count
                            next_city = neighbor

                if next_city != -1:
                    current = next_city
                    continue

            # No valid neighbors, pick random unused city
            unused_count = n - pos - 1
            if unused_count > 0:
                target = int(np.random.random() * unused_count)
                count = 0
                for i in range(n):
                    if not used[i]:
                        if count == target:
                            current = i
                            break
                        count += 1

    return offspring


# Modify the class name to match your student number.
class r0123456:
    def __init__(
        self,
        seed=None,
        recombination_type="pmx",
        tournament_k=46,
        population=2000,
        mutation_rate=0.09,
        mutation_scramble_prob=0.5,
    ):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

            # Store EA parameters as instance variables
        self.population_size = population
        self.tournament_k = tournament_k
        self.recombination_type = recombination_type  # 'edge' or 'pmx'
        self.mutation_rate = mutation_rate
        self.mutation_scramble_prob = (
            mutation_scramble_prob  # Probability of using scramble vs inversion
        )
        self.rng_state = np.random.randint(0, 2**32, size=4, dtype=np.uint32)

    @staticmethod
    @jit
    def init_population(pop_size: int, n_cities: int) -> np.ndarray:
        pop = np.empty((pop_size, n_cities), dtype=np.int32)
        base = np.arange(n_cities, dtype=np.int32)

        for i in range(pop_size):
            pop[i] = np.random.permutation(base)
        return pop

    def optimize(self, filename: str) -> float:
        distanceMatrix = np.loadtxt(filename, delimiter=",")

        n_cities = distanceMatrix.shape[0]
        pop_size = self.population_size
        max_iters = 10000

        # Initialize population and evaluate
        pop = self.init_population(pop_size, n_cities)
        fitness = _evaluate_population(pop, distanceMatrix)

        # Pre-allocate arrays for efficiency
        new_pop = np.empty_like(pop)
        parent_pairs = np.empty((pop_size - 1, 2), dtype=np.int32)

        it = 0
        no_improvement = 0
        best_ever = float("inf")
        best_solution = None

        while True:
            # Compute statistics once
            meanObjective = float(fitness.mean())
            print(meanObjective)
            best_idx = int(fitness.argmin())
            bestObjective = float(fitness[best_idx])

            # Update best solution if improved
            if bestObjective < best_ever:
                best_ever = bestObjective
                best_solution = pop[best_idx].copy()
                no_improvement = 0
            else:
                no_improvement += 1

            # Report progress using best known solution
            timeLeft = self.reporter.report(meanObjective, bestObjective, best_solution)
            if timeLeft < 0 or it >= max_iters or no_improvement >= 500:
                break

            # Pre-select all parent pairs for better cache efficiency
            parent_pairs = _select_parent_pairs(
                pop, fitness, self.tournament_k, pop_size - 1, self.rng_state
            )

            # Elitism: preserve best solution
            new_pop[0] = pop[best_idx]

            # Generate new population in batches for better vectorization
            for i in range(1, pop_size, 4):  # Process 4 solutions at a time
                batch_size = min(4, pop_size - i)
                for j in range(batch_size):
                    p1_idx, p2_idx = parent_pairs[i + j - 1]

                    # Recombination (using parent indices)
                    if self.recombination_type == "edge":
                        child = _edge_recombination(
                            pop[p1_idx], pop[p2_idx], self.rng_state
                        )
                    else:
                        child = pop[p1_idx].copy()

                    # Mutation (in-place when possible)
                    if np.random.random() < self.mutation_rate:
                        if np.random.random() < self.mutation_scramble_prob:
                            child = _scramble_mutation(
                                child, distanceMatrix, self.rng_state
                            )
                        else:
                            child = _inversion_mutation(
                                child, distanceMatrix, self.rng_state
                            )

                    new_pop[i + j] = child

            # Swap populations instead of copying
            pop, new_pop = new_pop, pop
            # Update fitness
            fitness = _evaluate_population(pop, distanceMatrix)
            it += 1

        return best_ever
