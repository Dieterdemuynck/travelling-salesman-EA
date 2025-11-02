import r0123456 as tsp
import threading
import optuna

# path = input("Enter path to TSP data file: ")
# path = "data/tour500.csv"
# tsp.r0123456().optimize(path)


def objective(trail: optuna.Trial):
    print(f"Running trial {trail.number=} in {threading.current_thread().name}")
    path = "data/tour500.csv"
    tournament_k = trail.suggest_int("tournament_k", 2, 100, log=True)
    population = trail.suggest_int("population", 100, 10000, step=100)
    mutation_rate = trail.suggest_float("mutation_rate", 0.001, 0.1)
    recomb = trail.suggest_categorical("recombination", ["edge", "pmx"])

    return tsp.r0123456(
        recombination_type=recomb,
        tournament_k=tournament_k,
        population=population,
        mutation_rate=mutation_rate,
        mutation_scramble_prob=0.5,
    ).optimize(
        path,
    )


study = optuna.create_study()
study.optimize(objective, n_trials=200, n_jobs=12)
print(study.best_params)
