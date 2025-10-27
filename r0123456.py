import Reporter
import numpy as np

# Modify the class name to match your student number.
class r0123456:

	def __init__(self):
		self.reporter = Reporter.Reporter(self.__class__.__name__)

	#Promt: I want to implement k-tournament selection for my evolutionary algorithm. Please do this by implementing a method, 'tournament_selection'. this method recieves a Numpy array, containing Numpy arrays, which represent paths in a traveling salesman problem. Each of these paths are permutations of indices in a distance matrix. the second argument is the evaluation method, which takes a path and returns a floating point number. this number represents the score of a given path. the third argument is another numpy arrays of numpy arrays, with k paths. These are filled with the found solutions after the tournament. make sure that after the function call, this argument is updated with the new solutions. 
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
		rng = np.random.default_rng()
		# Accept winners as numpy array or list-like; update in-place.
		for i in range(len(winners)):
			# choose k distinct competitors (or all if k >= pop_size)
			num_competitors = min(k, pop_size)
			indices = rng.choice(pop_size, size=num_competitors, replace=False)
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
