import Reporter
import unittest
import numpy as np

# Modify the class name to match your student number.
class r0123456:

	def __init__(self, seed = None):
		self.reporter = Reporter.Reporter(self.__class__.__name__)
		if seed is not None:
			self.rng = np.random.default_rng(seed)
		else:
			self.rng = np.random.default_rng()

	# k-tournament selection: fills `winners` (in-place) with selected solutions.
	def tournament_selection(self, population, evaluate, winners, k=3):
		"""
		Parameters:
		population : numpy.ndarray or sequence of numpy.ndarray
		Array-like of candidate solutions (each a 1D numpy array permutation).
		evaluate : callable
		Function taking a path (numpy array) and returning a float score (lower is better).
		winners : numpy.ndarray or sequence-like preallocated with slots for k selected paths.
		This will be updated in-place with the winners from each tournament.
		k : int
		Number of competitors per tournament.
		"""
		pop_size = len(population)
		if pop_size == 0:
			return
		if k <= 0:
			raise ValueError("k must be >= 1")
		# Accept winners as numpy array or list-like; update in-place.
		for i in range(len(winners)):
			# choose k distinct competitors (or all if k >= pop_size)
			num_competitors = min(k, pop_size)
			indices = self.rng.choice(pop_size, size=num_competitors, replace=False)
			best_idx = None
			best_score = None
			for idx in indices:
				cand = population[idx]
				score = evaluate(cand)
				if (best_score is None) or (score < best_score):
					best_score = score
					best_idx = idx
			# copy the winning candidate into the winners slot to avoid shared references
			win = np.array(population[best_idx], copy=True)
			# If winners is a numpy array of object dtype, assignment works; if it's a list, it also works.
			winners[i] = win


	# --- Mutation operators for permutations (TSP) ---

    # Scramble mutation: selects a subsequence and shuffles it.
	def scramble_mutation(self, solution, distanceMatrix):
		"""
		Parameters:
		solution : numpy.ndarray (1D permutation)
		distanceMatrix : 2D numpy.ndarray with distances; invalid edges are np.inf
		Returns:
		mutated solution (numpy.ndarray). If no valid mutation found, returns a copy of the original.
		"""
		n = solution.size
		if n <= 1:
			return np.array(solution, copy=True)
		def is_valid(path):
		# check all consecutive edges including return to start
			for i in range(n):
				a = path[i]
				b = path[(i+1) % n]
				if not np.isfinite(distanceMatrix[a, b]):
					return False
			return True
		# Try a number of random scramble attempts; if none valid, return original
		attempts = 20
		orig = np.array(solution, copy=True)
		for _ in range(attempts):
			i, j = sorted(self.rng.choice(n, size=2, replace=False))
			if i == j:
				continue
			sub = orig[i:j+1].copy()
			self.rng.shuffle(sub)
			cand = orig.copy()
			cand[i:j+1] = sub
			if is_valid(cand):
				return cand
		return orig

	# Inversion mutation: selects a subsequence and reverses it.
	def inversion_mutation(self, solution, distanceMatrix):
		"""
		Parameters:
		solution : numpy.ndarray (1D permutation)
		distanceMatrix : 2D numpy.ndarray with distances; invalid edges are np.inf
		Returns:
		mutated solution (numpy.ndarray). If no valid mutation found, returns a copy of the original.
		"""
		n = solution.size
		if n <= 1:
			return np.array(solution, copy=True)
		def is_valid(path):
			for i in range(n):
				a = path[i]
				b = path[(i+1) % n]
				if not np.isfinite(distanceMatrix[a, b]):
					return False
			return True
		attempts = 20
		orig = np.array(solution, copy=True)
		for _ in range(attempts):
			i, j = sorted(self.rng.choice(n, size=2, replace=False))
			if i == j:
				continue
			cand = orig.copy()
			cand[i:j+1] = cand[i:j+1][::-1]
			if is_valid(cand):
				return cand
		return orig



	@staticmethod
	def init_population(pop_size, n_cities, rng=None):
		rng = np.random.default_rng() if rng is None else rng
		pop = np.empty((pop_size, n_cities), dtype=int)
		base = np.arange(n_cities, dtype=int)
		# TODO: fixate first node, perhaps?
		for i in range(pop_size):
			pop[i] = rng.permutation(base)
		return pop

	@staticmethod
	def tour_length(route, dist):
		# route: 1D array of city indices, dist: 2D distance matrix
		# include return to start
		idx_from = route
		idx_to   = np.roll(route, -1)
		return dist[idx_from, idx_to].sum()

	@staticmethod
	def evaluate_population(pop, dist):
		# pop: (pop_size, n_cities)
		n = pop.shape[0]
		fitness = np.empty(n, dtype=float)
		for i in range(n):
			fitness[i] = r0123456.tour_length(pop[i], dist)
		return fitness

	# TODO: caching, perhaps?
	@staticmethod
	def build_edge_map(parent):
		# parent: 1D permutation -> list of neighbor sets
		n = parent.size
		edge = [set() for _ in range(n)]
		for i in range(n):
			a = parent[i]
			b = parent[(i-1)%n]
			c = parent[(i+1)%n]
			edge[a].update((b,c))
		return edge

	@staticmethod
	def edge_recombination(parent1, parent2, rng=None):
		"""
		Edge Recombination crossover producing one offspring from two parents.
		Implementation follows the standard ER: build adjacency from both parents,
		start from a random city, repeatedly choose next city among current's neighbors
		with minimal adjacency list size; if empty pick random remaining city.
		"""
		rng = np.random.default_rng() if rng is None else rng
		n = parent1.size
		# Build combined adjacency lists
		adj = [set() for _ in range(n)]
		e1 = r0123456.build_edge_map(parent1)
		e2 = r0123456.build_edge_map(parent2)
		for i in range(n):
			adj[i] = set(e1[i]) | set(e2[i])

		remaining = set(range(n))
		offspring = []
		# start city: prefer parents' first city or random
		start = int(rng.choice(parent1))  # pick a city from parent1 at random
		current = start

		while remaining:
			offspring.append(current)
			remaining.remove(current)
			# remove current from all adjacency lists
			for s in adj:
				s.discard(current)
			# choose next
			if remaining:
				neigh = adj[current] & remaining
				if neigh:
					# choose neighbor with smallest adjacency size (tie break randomly)
					cand = list(neigh)
					sizes = [len(adj[c]) for c in cand]
					min_size = min(sizes)
					min_cands = [c for c, sz in zip(cand, sizes) if sz == min_size]
					current = int(rng.choice(min_cands))
				else:
					# no neighbors left, choose uniformly from remaining
					current = int(rng.choice(list(remaining)))
		return np.array(offspring, dtype=int)

	def pmx(parent1, parent2):
		size = len(parent1)
		# Step 1: Choose two random crossover points
		cx1, cx2 = sorted(np.random.sample(range(size), 2))

		def create_offspring(p1, p2):
			offspring = [None] * size

			# Step 1: Copy the crossover segment from p1
			offspring[cx1:cx2] = p1[cx1:cx2]

			# Step 2: Handle elements in the crossover segment of p2 not yet in offspring
			for i in range(cx1, cx2):
				elem = p2[i]
				if elem not in offspring:
					pos = i
					while True:
						# Step 3: Find what element j was copied from p1 at this position
						j = p1[pos]
						# Step 4: Find where j is in p2
						pos = p2.index(j)
						# Step 5: If that position is empty, place elem there
						if offspring[pos] is None:
							offspring[pos] = elem
							break

			# Step 6: Fill remaining positions from p2
			for i in range(size):
				if offspring[i] is None:
					offspring[i] = p2[i]

			return offspring

		# Create both offspring
		child1 = create_offspring(parent1, parent2)
		child2 = create_offspring(parent2, parent1)

		return child1, child2

	# The evolutionary algorithm's main loop
	def optimize(self, filename):
		# Read distance matrix from file.		
		file = open(filename)
		distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()

		rng = np.random.default_rng()

		n_cities = distanceMatrix.shape[0]
		pop_size = 50
		max_iters = 10000

		pop = r0123456.init_population(pop_size, n_cities)
		fitness = r0123456.evaluate_population(pop, distanceMatrix)

		it = 0
		# Your code here.
		yourConvergenceTestsHere = True
		while( yourConvergenceTestsHere ):
			# compute statistics
			meanObjective = float(fitness.mean())
			best_idx = int(fitness.argmin())
			bestObjective = float(fitness[best_idx])
			bestSolution = pop[best_idx].copy()

			# Your code here.

			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution 
			#    with city numbering starting from 0
			timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
			if timeLeft < 0:
				break
			it += 1
			if it >= max_iters:
				break

			# Generate new population
			new_pop = np.empty_like(pop)
			# keep best individual
			new_pop[0] = bestSolution # TODO: better population selection

			for i in range(1, pop_size):
				# select two parents uniformly at random (without replacement)
				i1, i2 = rng.choice(pop_size, size=2, replace=False)
				p1 = pop[i1]
				p2 = pop[i2]

				# produce one or two children (we produce one here)
				child = r0123456.edge_recombination(p1, p2, rng=rng)
				# TODO: Mutation
				# optional small mutation: swap two non-first positions with tiny prob
				# if rng.random() < 0.05:
				# 	a = rng.integers(1, n_cities)  # avoid position 0 if fixed
				# 	b = rng.integers(1, n_cities)
				# 	child[a], child[b] = child[b], child[a]
				new_pop[i] = child
			pop = np.vstack(new_pop[:pop_size])
			fitness = r0123456.evaluate_population(pop, distanceMatrix)
		return 0

class TestEvolutionaryAlgorithm(unittest.TestCase):
	def test_inversion_operator(self):
		# seed == 0 -> i = 5, j = 7
		EA = r0123456(seed=0)
		amt_nodes = 8
		distance_matrix = np.array([[1.] * amt_nodes ] * amt_nodes)

		solution = EA.inversion_mutation(np.array([i for i in range(amt_nodes)]), distance_matrix)
		self.assertTrue(all(solution == np.array([0,1,2,3,4,7,6,5])), "The solutions must be equal!")
		
	def test_inversion_operator_but_inf_dist_once(self):
		EA = r0123456(seed=0)
		amt_nodes = 8
		distance_matrix = np.array([[1.] * amt_nodes ] * amt_nodes)
		# there is no link from 4 to 7.
		distance_matrix[4,7] = np.inf 

		# next choice due to invalid path = i = 1, j = 2
		solution = EA.inversion_mutation(np.array([i for i in range(amt_nodes)]), distance_matrix)
		self.assertTrue(all(solution == np.array([0,2,1,3,4,5,6,7])), "The solutions must be equal!")

	def test_scramble_operator(self):
		# seed == 0 -> i = 5, j = 7
		EA = r0123456(seed=0)
		amt_nodes = 8
		distance_matrix = np.array([[1.] * amt_nodes ] * amt_nodes)

		solution = EA.scramble_mutation(np.array([i for i in range(amt_nodes)]), distance_matrix)
		self.assertTrue(all(solution == np.array([0,1,2,3,4,6,7,5])), "The solutions must be equal!")

	def test_scramble_operator_but_inf_dist_once(self):
		EA = r0123456(seed=0)
		amt_nodes = 8
		distance_matrix = np.array([[1.] * amt_nodes ] * amt_nodes)
		# there is no link from 4 to 6.
		distance_matrix[4,6] = np.inf 

		# next choice due to invalid path = i = 0, j = 7
		solution = EA.scramble_mutation(np.array([i for i in range(amt_nodes)]), distance_matrix)
		self.assertTrue(all(solution == np.array([6, 2, 7, 4, 5, 1, 0, 3])), "The solutions must be equal!")

	def test_tournament_selection(self):
		EA = r0123456(seed=0)
		amt_nodes = 3
		amt_candidates = 10
		k = 3
		# generate random permutations.

		current_perm = [i for i in range(amt_nodes)]
		perms = np.array([current_perm])
		for i in range(amt_candidates):
			current_perm = EA.rng.permutation(current_perm)
			perms = np.append(perms, [current_perm], axis=0)

		def eval(path, distance_matrix = np.array([[1,2,3], [3,2,1], [2,3,1]])):
			fitness = 0.
			for i in range(len(path)):
				a = path[i]
				b = path[(i+1)%len(path)]
				fitness += distance_matrix[a,b]
			return fitness


		winners = [0] * k
		EA.tournament_selection(population=perms, evaluate=eval, k=k, winners=winners)
		self.assertTrue([all(i == j) for i,j in zip(np.array([[0, 1, 2], [0, 1, 2], [1, 2, 0]]), winners)])

