import Reporter
import numpy as np

# Modify the class name to match your student number.
class r0123456:

	def __init__(self):
		self.reporter = Reporter.Reporter(self.__class__.__name__)

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
			cx1, cx2 = sorted(random.sample(range(size), 2))

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
