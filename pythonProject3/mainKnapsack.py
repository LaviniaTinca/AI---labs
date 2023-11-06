import random
import time
import matplotlib.pyplot as plt


def parse_file_knapsack(filename):
    with open(filename, "r") as f:
        # Citim numarul de obiecte
        n = int(f.readline())
        # Citim obiectele
        items = []
        for i in range(n):
            linie = f.readline().split()
            index = int(linie[0])
            valoare = int(linie[2])
            greutate = int(linie[1])
            items.append((valoare, greutate))

        # Citim capacitatea rucsacului
        max_weight = int(f.readline())

    return max_weight, items

def generate_valid_solution(items, max_weight):
    while True:
        individual = [random.randint(0,1) for _ in range(len(items))]
        if isValid(individual, items, max_weight):
            return individual

def isValid(solution, items, max_weight):
    weight = 0
    for i in range(len(solution)):
        if solution[i] == 1:
            weight += items[i][1]
            if weight > max_weight:
                return False
    return True

# selectie turnir
def selection(population, fitnesses, tournament_size):
    tournament = random.sample(range(len(population)), tournament_size)
    best = tournament[0]
    for i in tournament:
        if fitnesses[i] > fitnesses[best]:
            best = i
    return population[best]

def crossover(parent1, parent2):
    # aleg un punct de incrucisare
    crossover_point = random.randint(0, len(parent1)-1)
    # incrucisarea propriu-zisa
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def uniform_crossover(parent1, parent2):
    child1 = []
    child2 = []
    for i in range(len(parent1)):
        if random.random() < 0.5:
            child1.append(parent1[i])
            child2.append(parent2[i])
        else:
            child1.append(parent2[i])
            child2.append(parent1[i])
    return child1, child2

# mutatie hard
def mutate(individual, mutation_probability):
    mutated = list(individual)
    for i in range(len(individual)):
        if random.random() < mutation_probability:
            mutated[i] = 1 - individual[i]
    return mutated

def mutate_swap(individual, mutation_probability):
    mutated = list(individual)
    if random.random() < mutation_probability:
        # Select two random indices
        i, j = random.sample(range(len(individual)), 2)
        # Swap the values at the selected indices
        mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated

def fitness(solution, items):
    value = 0
    for i in range(len(solution)):
        if solution[i] == 1:
            value += items[i][0]
    return value

# var2 incrucisare intr-un punct de taietura, mutatie hard (inversarea bitilor) pastram population_size
def evolutionary_algorithm_elite(population_size, num_generations, crossover_probability, mutation_probability, tournament_size, elite_percentage):
    population = []
    bestInGenerations =[]
    for i in range(population_size):
        individual = generate_valid_solution(items, max_weight)
        population.append(individual)
    start_time = time.time()
    for i in range(num_generations):
        # calculez fitness pt fiecare solutie si sortez descrescator (pr maximizare)
        fitnesses = [(fitness(solution, items), solution) for solution in population]
        fitnesses.sort(reverse=True)
        # pastrez elita generatiei anterioare
        elite_count = int(population_size * elite_percentage)
        new_population = [fitnesses[j][1] for j in range(elite_count)]
        # -------------------------------------------------

        # adaug noi copii prin incrucisare, mutatie
        children = []
        for j in range(elite_count, population_size):
            parent1 = selection(population, fitnesses, tournament_size)
            parent2 = selection(population, fitnesses, tournament_size)
            if random.random() < crossover_probability:
                child1, child2 = crossover(parent1, parent2)
                children.append(mutate(child1, mutation_probability))
                children.append(mutate(child2, mutation_probability))
            else:
                children.append(mutate(parent1, mutation_probability))
                children.append(mutate(parent2, mutation_probability))

        # selectam copiii si ii adaugam la noua populatie
        selected_children = selection_children(children,
                                               [(fitness(solution, items), solution) for solution in children],
                                               tournament_size)
        new_population = [fitnesses[j][1] for j in range(elite_count)]
        new_population.extend(selected_children)
        new_population = [solution for solution in new_population if isValid(solution, items, max_weight)]
        random.shuffle(new_population)

        # pastram dimensiunea populatiei si exemplarele bune
        new_population = sorted(new_population, key=lambda x: fitness(x, items), reverse=True)[:population_size]
        population = new_population
        #--------------------

        # calculez fitness si gasesc best solution in aceasta generatie
        fitnesses = [(fitness(solution, items), solution) for solution in population]
        fitnesses.sort(reverse=True)
        bestInGenerations.append(fitnesses[0][1])

        # print("Generation:", i+1, "Best solution:", fitnesses[0][1], "Fitness:", fitnesses[0][0])
    end_time = time.time()
    # determin cea mai buna solutie din generatii
    fitnesses = [(fitness(solution, items), solution) for solution in bestInGenerations]
    fitnesses.sort(reverse=True)
    best_solution = fitnesses[0][1]
    best_fitness = fitnesses[0][0]
    # print("Final population_size", len(population))
    return best_solution, best_fitness, end_time-start_time

def selection_children(children, fitnesses, tournament_size):
    selected = []
    while len(selected) < len(children):
        tournament = random.sample(fitnesses, tournament_size)
        winner = max(tournament, key=lambda x: x[0])[1]
        selected.append(winner)
    return selected

# var1 cu
def evolutionary_algorithm(population_size, num_generations, crossover_probability, mutation_probability, tournament_size):
    population = []
    bestInGenerations = []
    # generez aleator solutii valide pt populatie
    for i in range(population_size):
        solution = generate_valid_solution(items, max_weight)
        population.append(solution)
    start_time = time.time()
    for i in range(num_generations):
        fitnesses = [fitness(solution, items) for solution in population]
        new_population = []
        for j in range(population_size // 2):
            parent1 = selection(population, fitnesses, tournament_size)
            parent2 = selection(population, fitnesses, tournament_size)
            if random.random() < crossover_probability:
                child1, child2 = uniform_crossover(parent1, parent2)
                mutated_child1 = mutate_swap(child1, mutation_probability)
                mutated_child2 = mutate_swap(child2, mutation_probability)
                if isValid(mutated_child1, items, max_weight):
                    new_population.append(mutated_child1)
                if isValid(mutated_child2, items, max_weight):
                    new_population.append(mutated_child2)
            else:
                mutated_parent1 = mutate_swap(parent1, mutation_probability)
                mutated_parent2 = mutate_swap(parent2, mutation_probability)
                if isValid(mutated_parent1, items, max_weight):
                    new_population.append(mutated_parent1)
                if isValid(mutated_parent2, items, max_weight):
                    new_population.append(mutated_parent2)
        population = new_population
        # calculez fitness si gasesc best solution in aceasta generatie
        fitnesses = [(fitness(solution, items), solution) for solution in population]
        fitnesses.sort(reverse=True)
        bestInGenerations.append(fitnesses[0][1])
        # print("Generation:", i+1, "Best solution:", fitnesses[0][1], "Fitness:", fitnesses[0][0])

        # # Calculate fitness for each individual in the population
        # fitnesses = [fitness(solution, items) for solution in population]
        #
        # # Find the best solution in the current generation
        # best_solution = population[fitnesses.index(max(fitnesses))]
        # best_fitness = max(fitnesses)
        #
        # # Print the best solution in the current generation
        # print(f"Generation {i + 1}: Best solution: {best_solution}, Best fitness: {best_fitness}")
    end_time = time.time()
    # best_solution = population[fitnesses.index(max(fitnesses))]
    # # print("Final population_size", len(population))
    # return best_solution, max(fitnesses), end_time-start_time

    # determin cea mai buna solutie din generatii
    fitnesses = [(fitness(solution, items), solution) for solution in bestInGenerations]
    fitnesses.sort(reverse=True)
    best_solution = fitnesses[0][1]
    best_fitness = fitnesses[0][0]
    # print("Final population_size", len(population))
    return best_solution, best_fitness, end_time - start_time

# var1 cu incrucisare uniforma, mutate_swap si noua generatie formata din descendenti
def evolutionary_algorithm_samePopulationSize(population_size, num_generations, crossover_probability, mutation_probability, tournament_size):
    population = []
    bestInGenerations = []
    # generez aleator solutii valide pt populatie
    for i in range(population_size):
        solution = generate_valid_solution(items, max_weight)
        population.append(solution)
    start_time = time.time()
    for i in range(num_generations):
        fitnesses = [fitness(solution, items) for solution in population]
        new_population = []
        for j in range(population_size // 2):
            while len(new_population) < population_size:
                parent1 = selection(population, fitnesses, tournament_size)
                parent2 = selection(population, fitnesses, tournament_size)
                if random.random() < crossover_probability:
                    child1, child2 = uniform_crossover(parent1, parent2)
                    mutated_child1 = mutate_swap(child1, mutation_probability)
                    mutated_child2 = mutate_swap(child2, mutation_probability)
                    if isValid(mutated_child1, items, max_weight):
                        new_population.append(mutated_child1)
                    if isValid(mutated_child2, items, max_weight):
                        new_population.append(mutated_child2)
                else:
                    mutated_parent1 = mutate_swap(parent1, mutation_probability)
                    mutated_parent2 = mutate_swap(parent2, mutation_probability)
                    if isValid(mutated_parent1, items, max_weight):
                        new_population.append(mutated_parent1)
                    if isValid(mutated_parent2, items, max_weight):
                        new_population.append(mutated_parent2)
        population = new_population
        # calculez fitness si gasesc best solution in aceasta generatie
        fitnesses = [(fitness(solution, items), solution) for solution in population]
        fitnesses.sort(reverse=True)
        bestInGenerations.append(fitnesses[0][1])
        # print("Generation:", i+1, "Best solution:", fitnesses[0][1], "Fitness:", fitnesses[0][0])
    end_time = time.time()
    # determin cea mai buna solutie din generatii
    fitnesses = [(fitness(solution, items), solution) for solution in bestInGenerations]
    fitnesses.sort(reverse=True)
    best_solution = fitnesses[0][1]
    best_fitness = fitnesses[0][0]
    return best_solution, best_fitness, end_time - start_time

def result(file, population_size, num_generations, crossover_probability,
                                       mutation_probability, tournament_size, elite_percentage):
    with open(file, 'w') as file:
        solutions_knapsack = []
        solutions_knapsack_elite = []
        print(f'Pentru problema rucsacului avem n = {len(items)}\n'
              f'Capacitate rucsac = {max_weight}\n', file=file)
        for i in range(1, 11):
            print(f' Iteratia {i}: \n', file=file)
            solution1 = evolutionary_algorithm_samePopulationSize(population_size, num_generations, crossover_probability, mutation_probability, tournament_size)
            solutions_knapsack.append(solution1)
            print(
                f'Pentru problema rucsacului avem solutia: {solution1[0]}\n cu fitness: {solution1[1]} \n timp de executie {solution1[2]}\n',
                file=file)
            solution2 = evolutionary_algorithm_elite(population_size, num_generations, crossover_probability, mutation_probability, tournament_size, elite_percentage)

            solutions_knapsack_elite.append(solution2)
            print(
                f'Pentru problema Knapsack elite avem solutia: {solution2[0]}\n cu fitness:  {solution2[1]}\n timp de executie {solution2[2]}\n',
                file=file)

        iteratii = list(range(1, 11))
        fitness = []
        sum =0
        # fitness_values2 = []
        for sol in solutions_knapsack_elite:
            fitness.append(sol[1])
            sum+=sol[1]
        plt.plot(iteratii, fitness)
        # plt.plot(iteratii, fitness_values2)
        # plt.bar(iteratii, fitness_values)
        plt.title('Knapsack 20 BEST')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.show()
        # for sol in solutions_tsp1:
        #     fitness_values1.append(sol[1])
        # # for sol in fitness_values:
        # #     print(sol)

        print(f'------Sumar------\n\n'
              f'-- Avem {len(items)} elemente in rucsac\n'
              f'--population_size {population_size}, num_generations {num_generations}, crossover_probability {crossover_probability}, mutation_probability {mutation_probability}, tournament_size {tournament_size}, elite_percentage {elite_percentage} \n'
              f' - AE pentru problema rucsacului: (var 1 - incrucisare uniforma, generatii noi fara elitele anterioare)  \n', file = file)
        print_sumar(file, solutions_knapsack, maxim(solutions_knapsack), minim(solutions_knapsack))
        print(f' - Elite AE pentru problema rucsacului: (var 2 - incrucisare intr-un punct de taietura, cu elitele anterioare) \n', file=file)
        print_sumar(file, solutions_knapsack_elite, maxim(solutions_knapsack_elite), minim(solutions_knapsack_elite))

def print_sumar(file, solutii, best, worst):
    print(f'cea mai buna solutie a fost: {solutii[best[1]][0]},\n calitate = {best[0]}\n', file=file)
    print(f'cea mai slaba solutie a fost: {solutii[worst[1]][0]},\n calitate = {worst[0]}\n', file=file)
    total = 0
    timpi = 0
    for sol in solutii:
        total += sol[1]
        timpi += sol[2]
    print(f'calitatea medie:  {total / len(solutii)}\n', file=file)
    print(f'timp mediu de executie: {timpi / len(solutii)}\n', file=file)

def maxim(solutii):
    max = 0
    index = 0
    for i in range(len(solutii)):
        if max < solutii[i][1]:
            max = solutii[i][1]
            index = i
    return max, index


def minim(solutii):
    min, index = maxim(solutii)
    for i in range(len(solutii)):
        if min > solutii[i][1]:
            min = solutii[i][1]
            index = i
    return min, index

###############################################
# knapsack

k_file = "rucsac-20.txt"
max_weight, items = parse_file_knapsack(k_file)
# population_size = int(input('Marime populatie:  '))
# num_generations = int(input('Nr generatii:  '))
# crossover_probability = int(input('Probabilitate de incrucisare <1:  '))
# mutation_probability = int(input('Probabilitate de mutatie <1:  '))
# elite_percentage = int(input('Procentul elitei <1:  '))
# rulam algoritmul evolutiv elitist cu parametrii specificati


population_size=100
num_generations=30
crossover_probability=0.5
mutation_probability=0.5
tournament_size = 2
elite_percentage = 0.1
result_file = "zknapsack-test.txt"
result(result_file, population_size, num_generations, crossover_probability,
                                       mutation_probability, tournament_size, elite_percentage)
# best_solution, best_fitness, run_time = evolutionary_algorithm_elite(population_size, num_generations, crossover_probability, mutation_probability, tournament_size, elite_percentage)
# best_solution, best_fitness, run_time = evolutionary_algorithm(population_size, num_generations, crossover_probability, mutation_probability, tournament_size)
# best_solution, best_fitness, run_time = evolutionary_algorithm_samePopulationSize(population_size, num_generations, crossover_probability, mutation_probability, tournament_size)
# print(f"Best solution: {best_solution}")
# print(f"Best fitness: {best_fitness}")
# print(f"Run time: {run_time} seconds")