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



	# The evolutionary algorithm's main loop
	def optimize(self, filename):
		# Read distance matrix from file.		
		file = open(filename)
		distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()

		# Your code here.
		yourConvergenceTestsHere = True
		while( yourConvergenceTestsHere ):
			meanObjective = 0.0
			bestObjective = 0.0
			bestSolution = np.array([1,2,3,4,5])

			# Your code here.

			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution 
			#    with city numbering starting from 0
			timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
			if timeLeft < 0:
				break

		# Your code here.
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
