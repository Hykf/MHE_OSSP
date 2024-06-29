import numpy as np
from copy import deepcopy
import random
from itertools import permutations
from concurrent.futures import ThreadPoolExecutor
import argparse

offest = 1e-6 ##TODO wywalic smutne obejscie

def johnson_rule(processing_times):

    #https://www.geeksforgeeks.org/johnsons-rule-in-sequencing-problems/
    #https://en.wikipedia.org/wiki/Johnson%27s_rule

    jobs = list(range(processing_times.shape[0]))
    schedule = []

    while jobs:
        if len(jobs) == 1:
            schedule.append(jobs[0])
            break

        min_time = np.min(processing_times[jobs])
        min_jobs = np.where(processing_times[jobs] == min_time)[0]

        if any(processing_times[jobs[i], :1] == min_time for i in min_jobs):
            schedule.insert(0, jobs.pop(min_jobs[0]))
        else:
            schedule.append(jobs.pop(min_jobs[-1]))

    return schedule

def calculate_makespan(processing_times, schedule):
    num_machines = processing_times.shape[1]
    completion_times = np.zeros((len(schedule), num_machines))

    for j in range(num_machines):
        for i in range(len(schedule)):
            if i == 0:
                completion_times[i, j] = processing_times[schedule[i], j]
            else:
                completion_times[i, j] = max(completion_times[i - 1, j], completion_times[i, j - 1]) + processing_times[schedule[i], j]

    makespan = completion_times[-1, -1]
    return makespan, completion_times

def objective_function(processing_times, schedule, completion_times):

    makespan = completion_times[-1, -1]  # Pobieramy makespan z completion_times
    num_machines = processing_times.shape[1]
    max_makespan = np.sum(processing_times)

    idle_times = np.zeros(num_machines)
    for j in range(num_machines):
        for i in range(1, len(schedule)):
            idle_times[j] += max(0, completion_times[i, j - 1] - completion_times[i - 1, j])

    total_idle_time = np.sum(idle_times)
    max_idle_time = max_makespan * (num_machines - 1)

    normalized_idle_time = total_idle_time / max_idle_time

    return 1 - normalized_idle_time

def generate_neighborhood(schedule):

    neighborhood = []
    for i in range(len(schedule) - 1):
        for j in range(i + 1, len(schedule)):
            neighbor = deepcopy(schedule)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighborhood.append(neighbor)
    return neighborhood
def generate_random_solution(num_jobs):

    jobs = list(range(num_jobs))
    random.shuffle(jobs)
    return jobs

def brute_force_search(processing_times):

    num_jobs = processing_times.shape[0]
    all_solutions = []

    for schedule in permutations(range(num_jobs)):
        makespan, completion_times = calculate_makespan(processing_times, schedule)
        score = objective_function(processing_times, schedule, completion_times)
        all_solutions.append((schedule, makespan, score))

    return all_solutions
def hill_climbing(processing_times, initial_solution):

    current_solution = initial_solution
    current_makespan, current_completion_times = calculate_makespan(processing_times, current_solution)
    current_score = objective_function(processing_times, current_solution, current_completion_times)

    while True:
        neighborhood = generate_neighborhood(current_solution)
        best_neighbor = None
        best_neighbor_score = current_score

        for neighbor in neighborhood:
            neighbor_makespan, neighbor_completion_times = calculate_makespan(processing_times, neighbor)
            neighbor_score = objective_function(processing_times, neighbor, neighbor_completion_times)
            if neighbor_score < best_neighbor_score:
                best_neighbor = neighbor
                best_neighbor_score = neighbor_score

        if best_neighbor_score >= current_score:
            break

        current_solution = best_neighbor
        current_score = best_neighbor_score
        current_completion_times = neighbor_completion_times

    return current_solution, current_score
def hill_climbing_random_neighbor(processing_times, initial_solution):

    current_solution = initial_solution
    current_makespan, current_completion_times = calculate_makespan(processing_times, current_solution)
    current_score = objective_function(processing_times, current_solution, current_completion_times)

    while True:
        neighborhood = generate_neighborhood(current_solution)
        random_neighbor = random.choice(neighborhood)  # Wybieramy losowego sąsiada

        neighbor_makespan, neighbor_completion_times = calculate_makespan(processing_times, random_neighbor)
        neighbor_score = objective_function(processing_times, random_neighbor, neighbor_completion_times)

        if neighbor_score < current_score:  # Jeśli losowy sąsiad jest lepszy, akceptujemy go
            current_solution = random_neighbor
            current_score = neighbor_score
            current_completion_times = neighbor_completion_times
        else:
            break  # Jeśli nie znaleźliśmy lepszego sąsiada, kończymy

    return current_solution, current_score

#https://en.wikipedia.org/wiki/Tabu_search
def tabu_search(processing_times, initial_solution, tabu_size=10, max_iterations=100):

    current_solution = initial_solution
    best_solution = initial_solution
    tabu_list = []
    best_makespan, best_completion_times = calculate_makespan(processing_times, best_solution)
    best_score = objective_function(processing_times, best_solution, best_completion_times)

    for _ in range(max_iterations):
        neighborhood = generate_neighborhood(current_solution)
        best_neighbor = None
        best_neighbor_score = float('inf')

        for neighbor in neighborhood:
            if neighbor not in tabu_list:  # Sprawdzamy, czy sąsiad nie jest na liście tabu
                neighbor_makespan, neighbor_completion_times = calculate_makespan(processing_times, neighbor)
                neighbor_score = objective_function(processing_times, neighbor, neighbor_completion_times)
                if neighbor_score < best_neighbor_score:
                    best_neighbor = neighbor
                    best_neighbor_score = neighbor_score

        # Jeśli nie znaleźliśmy lepszego sąsiada spoza listy tabu, wybieramy najlepszego z listy tabu
        if best_neighbor is None:
            best_neighbor = min(neighborhood, key=lambda x: objective_function(processing_times, x))

        current_solution = best_neighbor
        if best_neighbor_score < best_score:
            best_solution = best_neighbor
            best_score = best_neighbor_score
            best_completion_times = neighbor_completion_times

        tabu_list.append(current_solution)  # Dodajemy bieżące rozwiązanie do listy tabu
        if len(tabu_list) > tabu_size:  # Usuwamy najstarszy element z listy tabu, jeśli przekroczyła rozmiar
            tabu_list.pop(0)

    return best_solution, best_score

def tabu_search_with_backtracking(processing_times, initial_solution, tabu_size=10, max_iterations=100):

    current_solution = initial_solution
    best_solution = initial_solution
    tabu_list = []
    best_makespan, best_completion_times = calculate_makespan(processing_times, best_solution)
    best_score = objective_function(processing_times, best_solution, best_completion_times)

    for _ in range(max_iterations):
        neighborhood = generate_neighborhood(current_solution)
        best_neighbor = None
        best_neighbor_score = float('inf')

        for neighbor in neighborhood:
            if neighbor not in tabu_list:
                neighbor_makespan, neighbor_completion_times = calculate_makespan(processing_times, neighbor)
                neighbor_score = objective_function(processing_times, neighbor, neighbor_completion_times)
                if neighbor_score < best_neighbor_score:
                    best_neighbor = neighbor
                    best_neighbor_score = neighbor_score

        # Jeśli nie znaleźliśmy lepszego sąsiada spoza listy tabu, cofamy się na liście tabu
        if best_neighbor is None:
            for solution in reversed(tabu_list):
                if solution not in neighborhood:
                    best_neighbor = solution
                    break

        current_solution = best_neighbor
        if best_neighbor_score < best_score:
            best_solution = best_neighbor
            best_score = best_neighbor_score
            best_completion_times = neighbor_completion_times

        tabu_list.append(current_solution)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

    return best_solution, best_score

# https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)
def one_point_crossover(parent1, parent2):

    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + [x for x in parent2 if x not in parent1[:crossover_point]]
    child2 = parent2[:crossover_point] + [x for x in parent1 if x not in parent2[:crossover_point]]
    return child1, child2
def order_crossover(parent1, parent2):

    start, end = sorted(random.sample(range(len(parent1)), 2))
    child1 = [-1] * len(parent1)
    child1[start:end+1] = parent1[start:end+1]
    child2 = [-1] * len(parent2)
    child2[start:end+1] = parent2[start:end+1]

    remaining1 = [x for x in parent2 if x not in child1]
    remaining2 = [x for x in parent1 if x not in child2]

    for i in range(len(parent1)):
        if child1[i] == -1:
            child1[i] = remaining1.pop(0)
        if child2[i] == -1:
            child2[i] = remaining2.pop(0)

    return child1, child2

# https://en.wikipedia.org/wiki/Mutation_(genetic_algorithm)
def swap_mutation(individual, mutation_probability=0.1):

    mutated = individual.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_probability:
            j = random.randint(0, len(mutated) - 1)
            mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated

def inversion_mutation(individual, mutation_probability=0.1):
    mutated = individual.copy()
    if random.random() < mutation_probability:
        start, end = sorted(random.sample(range(len(mutated)), 2))
        mutated[start:end+1] = reversed(mutated[start:end+1])
    return mutated

def genetic_algorithm(processing_times, population_size=50, crossover_method="one_point", mutation_method="swap", termination_condition="iterations", max_iterations=100, no_improvement_limit=20):

    population = [generate_random_solution(processing_times.shape[0]) for _ in range(population_size)]
    population_scores = [objective_function(processing_times, individual, calculate_makespan(processing_times, individual)[1]) for individual in population]

    best_solution = population[np.argmin(population_scores)]
    best_score = min(population_scores)
    no_improvement_counter = 0

    for iteration in range(max_iterations):
        # Selekcja
        parents = random.choices(population, weights=[1/(score+offest) for score in population_scores], k=population_size)  # Selekcja proporcjonalna do fitness (1/score)

        # Krzyżowanie
        new_population = []
        for i in range(0, population_size, 2):
            parent1, parent2 = parents[i], parents[i+1]
            if crossover_method == "one_point":
                child1, child2 = one_point_crossover(parent1, parent2)
            else:  # crossover_method == "order"
                child1, child2 = order_crossover(parent1, parent2)
            new_population.extend([child1, child2])

        # Mutacja
        for i in range(population_size):
            if mutation_method == "swap":
                new_population[i] = swap_mutation(new_population[i])
            else:  # mutation_method == "inversion"
                new_population[i] = inversion_mutation(new_population[i])

        new_population_scores = [objective_function(processing_times, individual, calculate_makespan(processing_times, individual)[1]) for individual in new_population]

        if min(new_population_scores) < best_score:
            best_solution = new_population[np.argmin(new_population_scores)]
            best_score = min(new_population_scores)
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1

        # Warunki zakończenia
        if termination_condition == "iterations" and iteration >= max_iterations - 1:
            break
        elif termination_condition == "no_improvement" and no_improvement_counter >= no_improvement_limit:
            break

        population = new_population
        population_scores = new_population_scores

    return best_solution, best_score

def genetic_algorithm_parallel(processing_times, population_size=50, crossover_method="one_point", mutation_method="swap", termination_condition="iterations", max_iterations=100, no_improvement_limit=20):

    population = [generate_random_solution(processing_times.shape[0]) for _ in range(population_size)]
    best_solution = None
    best_score = float('inf')

    with ThreadPoolExecutor() as executor:
        for iteration in range(max_iterations):
            population_scores = list(executor.map(evaluate_individual, population, [processing_times] * population_size))
            parents = random.choices(population, weights=[1 /(score+offest) for score in population_scores], k=population_size)

            # Krzyżowanie
            new_population = []
            for i in range(0, population_size, 2):
                parent1, parent2 = parents[i], parents[i + 1]
                if crossover_method == "one_point":
                    child1, child2 = one_point_crossover(parent1, parent2)
                else:  # crossover_method == "order"
                    child1, child2 = order_crossover(parent1, parent2)
                new_population.extend([child1, child2])

            # Mutacja
            for i in range(population_size):
                if mutation_method == "swap":
                    new_population[i] = swap_mutation(new_population[i])
                else:  # mutation_method == "inversion"
                    new_population[i] = inversion_mutation(new_population[i])

            new_population_scores = list(executor.map(evaluate_individual, new_population, [processing_times] * population_size))

            best_index = np.argmin(new_population_scores)
            if new_population_scores[best_index] < best_score:
                best_solution = new_population[best_index]
                best_score = new_population_scores[best_index]
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            if termination_condition == "iterations" and iteration >= max_iterations - 1:
                break
            elif termination_condition == "no_improvement" and no_improvement_counter >= no_improvement_limit:
                break

            population = new_population

    return best_solution, best_score

def evaluate_individual(individual, processing_times):

    makespan, completion_times = calculate_makespan(processing_times, individual)
    return objective_function(processing_times, individual, completion_times)

############## - MAIN - #############################
if __name__ == "__main__":

    # Parsowanie argumentów wiersza poleceń
    parser = argparse.ArgumentParser(description="Algorytm genetyczny dla problemu OSSP")

    parser.add_argument("--crossover_method", type=str, default="one_point", choices=["one_point", "order"], help="Metoda krzyżowania")
    parser.add_argument("--mutation_method", type=str, default="swap", choices=["swap", "inversion"], help="Metoda mutacji")
    parser.add_argument("--termination_condition", type=str, default="iterations", choices=["iterations", "no_improvement"], help="Warunek zakończenia")

    parser.add_argument("--population_size", type=int, default=50, help="Rozmiar populacji")
    parser.add_argument("--max_iterations", type=int, default=100, help="Maksymalna liczba iteracji")
    parser.add_argument("--no_improvement_limit", type=int, default=20, help="Limit iteracji bez poprawy")
    parser.add_argument("--tabu_size", type=int, default=10, help="Rozmiar listy tabu")
    parser.add_argument("--num_jobs", type=int, default=10, help="Ilosc zadan w macierzy")

    parser.add_argument("--brute_force", type=bool, default=False, help="Wyswietlenie bruteforce")
    args = parser.parse_args()

    max_machines = 4
    max_processing_time = 10
    num_jobs = args.num_jobs

    num_machines = random.randint(3, max_machines)
    processing_times = np.random.randint(1, max_processing_time + 1, size=(num_jobs, num_machines))

    print("Wylosowane czasy przetwarzania:")
    print(processing_times)

    ######### - INIT #######################################

    # Rozwiązanie problemu dla wszystkich trzech maszyn
    schedule_3_machines = johnson_rule(np.vstack((processing_times[:, 0], processing_times[:, 2])).T)
    makespan_3_machines, completion_times_3_machines = calculate_makespan(processing_times, schedule_3_machines)
    objective_value_3_machines = objective_function(processing_times, schedule_3_machines, completion_times_3_machines)
    print("\nWejście  :", processing_times)
    print("Wynik  :", schedule_3_machines)
    print("Makespan  :", makespan_3_machines)
    print("Score  :", objective_value_3_machines)

    # Generowanie losowego rozwiązania dla 3 maszyn i ocena
    random_solution = generate_random_solution(processing_times.shape[0])
    random_makespan, random_completion_times = calculate_makespan(processing_times, random_solution) # Oblicz makespan i completion_times
    random_score = objective_function(processing_times, random_solution, random_completion_times)
    print("\nLosowe rozwiązanie  :", random_solution)
    print("Makespan:", random_makespan)
    print("Score:", random_score)

    # Generowanie i ocena sąsiedztwa dla harmonogramu 3 maszyn
    neighborhood_3_machines = generate_neighborhood(schedule_3_machines)
    for neighbor in neighborhood_3_machines:
        neighbor_makespan, neighbor_completion_times = calculate_makespan(processing_times, neighbor) # Oblicz makespan i completion_times
        neighbor_score = objective_function(processing_times, neighbor, neighbor_completion_times) # Przekazujemy completion_times
        print("Sąsiad  :", neighbor, "Makespan:", neighbor_makespan, "Score:", neighbor_score)

    if args.brute_force:
        # Rozwiązanie problemu dla 3 maszyn za pomocą pełnego przeglądu
        all_solutions_3_machines = brute_force_search(processing_times)

        print("\nPełny przegląd  :")
        for schedule, makespan, score in all_solutions_3_machines:
            print("Harmonogram:", schedule, "Makespan:", makespan, "Score:", score)

    # Rozwiązanie problemu dla 3 maszyn za pomocą algorytmu wspinaczkowego
    initial_solution = generate_random_solution(processing_times.shape[0])
    best_schedule_hill_climbing, best_makespan_hill_climbing = hill_climbing(processing_times, initial_solution)
    print("\nAlgorytm wspinaczkowy  :")
    print("Najlepszy harmonogram:", best_schedule_hill_climbing)
    print("Najlepszy makespan:", best_makespan_hill_climbing)

    # Rozwiązanie problemu dla 3 maszyn za pomocą algorytmu wspinaczkowego z losowym wyborem sąsiada
    initial_solution = generate_random_solution(processing_times.shape[0])
    best_schedule_hill_climbing_random, best_makespan_hill_climbing_random = hill_climbing_random_neighbor(processing_times, initial_solution)

    print("\nAlgorytm wspinaczkowy z losowym wyborem sąsiada  :")
    print("Najlepszy harmonogram:", best_schedule_hill_climbing_random)
    print("Najlepszy makespan:", best_makespan_hill_climbing_random)

    # Rozwiązanie problemu dla 3 maszyn za pomocą algorytmu Tabu Search
    initial_solution = generate_random_solution(processing_times.shape[0])
    tabu_size = args.tabu_size
    best_schedule_tabu, best_makespan_tabu = tabu_search(processing_times, initial_solution, tabu_size)

    print("\nAlgorytm Tabu Search :")
    print("Najlepszy harmonogram:", best_schedule_tabu)
    print("Najlepszy makespan:", best_makespan_tabu)

    # Rozwiązanie problemu dla 3 maszyn za pomocą algorytmu Tabu Search z cofaniem się
    initial_solution = generate_random_solution(processing_times.shape[0])
    tabu_size = args.tabu_size
    best_schedule_tabu, best_makespan_tabu = tabu_search_with_backtracking(processing_times, initial_solution, tabu_size)

    print("\nAlgorytm Tabu Search z cofaniem się:")
    print("Najlepszy harmonogram:", best_schedule_tabu)
    print("Najlepszy makespan:", best_makespan_tabu)

    # Krzyżowanie
    parent1 = generate_random_solution(processing_times.shape[0])
    parent2 = generate_random_solution(processing_times.shape[0])

    child1, child2 = one_point_crossover(parent1, parent2)
    makespan_child1, completion_times_child1 = calculate_makespan(processing_times, child1)
    score_child1 = objective_function(processing_times, child1, completion_times_child1)
    makespan_child2, completion_times_child2 = calculate_makespan(processing_times, child2)
    score_child2 = objective_function(processing_times, child2, completion_times_child2)
    print(f"\nDzieci po krzyżowaniu jednopunktowym: {child1} (makespan: {makespan_child1}, score: {score_child1}), {child2} (makespan: {makespan_child2}, score: {score_child2})")

    child1, child2 = order_crossover(parent1, parent2)
    makespan_child1, completion_times_child1 = calculate_makespan(processing_times, child1)
    score_child1 = objective_function(processing_times, child1, completion_times_child1)
    makespan_child2, completion_times_child2 = calculate_makespan(processing_times, child2)
    score_child2 = objective_function(processing_times, child2, completion_times_child2)
    print(f"Dzieci po krzyżowaniu uporządkowanym: {child1} (makespan: {makespan_child1}, score: {score_child1}), {child2} (makespan: {makespan_child2}, score: {score_child2})")

    # Przykład użycia mutacji
    parent = generate_random_solution(processing_times.shape[0])
    print("\nRodzic:", parent)

    mutated_child1 = swap_mutation(parent)
    makespan_child1, completion_times_child1 = calculate_makespan(processing_times, mutated_child1)
    score_child1 = objective_function(processing_times, mutated_child1, completion_times_child1)
    print(f"\nDziecko po mutacji inwersji:: {mutated_child1} \n(makespan: {makespan_child1}, score: {score_child1})")


    mutated_child2 = inversion_mutation(parent)
    makespan_child2, completion_times_child2 = calculate_makespan(processing_times, mutated_child2)
    score_child2 = objective_function(processing_times, mutated_child2, completion_times_child2)
    print(f"\nDziecko po mutacji inwersji:: {mutated_child2} \n(makespan: {makespan_child2}, score: {score_child2})")


    best_schedule_ga, best_score_ga = genetic_algorithm(
        processing_times,
        population_size=args.population_size,
        crossover_method=args.crossover_method,
        mutation_method=args.mutation_method,
        termination_condition=args.termination_condition,
        no_improvement_limit=args.no_improvement_limit,
    )

    print("\nAlgorytm genetyczny:")
    print("Najlepszy harmonogram:", best_schedule_ga)
    print("Najlepszy makespan:", calculate_makespan(processing_times, best_schedule_ga)[0])
    print("Najlepszy score:", best_score_ga)

    # Rozwiązanie problemu dla 3 maszyn za pomocą zrównoleglonego algorytmu genetycznego
    best_schedule_ga_parallel, best_score_ga_parallel = genetic_algorithm_parallel(
        processing_times,
        population_size=args.population_size,
        crossover_method=args.crossover_method,
        mutation_method=args.mutation_method,
        termination_condition=args.termination_condition,
        no_improvement_limit=args.no_improvement_limit,
    )
    print("\nRównoleglony algorytm genetyczny:")
    print("Najlepszy harmonogram:", best_schedule_ga_parallel)
    print("Najlepszy makespan:", calculate_makespan(processing_times, best_schedule_ga_parallel)[0])
    print("Najlepszy score:", best_score_ga_parallel)
