import random
import time
import matplotlib.pyplot as plt


# definesc functia de fitness pentru a calcula distanta (merg pe varianta de maximizare -> returnez inversul)
def fitness(solution, cities):
    distance = 0
    for i in range(len(solution)):
        city1 = solution[i]
        city2 = solution[(i+1)%len(solution)]
        x1, y1 = cities[city1]
        x2, y2 = cities[city2]
        distance += ((x1-x2)**2 + (y1-y2)**2) ** 0.5
    return 1/distance

# generez solutie
def generate_valid_solution(num_cities):
    solution = list(range(num_cities))
    random.shuffle(solution)
    return solution

# selectia turnir pentru parinti
def selection(population, fitnesses, tournament_size):
    tournament = random.sample(fitnesses, tournament_size)
    tournament.sort(reverse=True)
    return tournament[0][1]

# incrucisare PMX partially mapped crossover intre doi parinti
def pmx_crossover(parent1, parent2):
    # aleg un punct de incrucisare
    crossover_point = random.randint(1, len(parent1)-1)
    child1 = parent1[:crossover_point] + [city for city in parent2 if city not in parent1[:crossover_point]]
    child2 = parent2[:crossover_point] + [city for city in parent1 if city not in parent2[:crossover_point]]
    return child1, child2


# incrucisare ordered crossover intre doi parinti
def ordered_crossover(parent1, parent2):
    # aleg 2 puncte
    crossover_points = sorted(random.sample(range(len(parent1)), 2))
    child1 = [-1] * len(parent1)
    child2 = [-1] * len(parent1)

    # copilul1 preia un set de gene din parintele 1
    subset = parent1[crossover_points[0]:crossover_points[1] + 1]
    child1[crossover_points[0]:crossover_points[1] + 1] = subset

    # continui cu genele din parinte2 pastrand ordinea
    j = crossover_points[1] + 1
    for i in range(j, j + len(parent2)):
        if parent2[i % len(parent2)] not in subset:
            child1[j % len(parent1)] = parent2[i % len(parent2)]
            j += 1

    # schimb rolurile parintilor pt a genera copil2
    child2, parent1 = parent1[:], parent2[:]
    j = crossover_points[1] + 1
    for i in range(j, j + len(parent1)):
        if parent1[i % len(parent1)] not in subset:
            child2[j % len(parent2)] = parent1[i % len(parent1)]
            j += 1

    return child1, child2

# mutatie
def mutate_swap(solution, mutation_probability):
    if random.random() < mutation_probability:
        # Swap 2 orase
        index1 = random.randint(0, len(solution)-1)
        index2 = random.randint(0, len(solution)-1)
        solution[index1], solution[index2] = solution[index2], solution[index1]
    return solution

# AE pmx crossorder, mutate_swap
def evolutionary_algorithm_elite1(population_size, num_generations, crossover_probability, mutation_probability, tournament_size, elite_percentage):
    population = []
    bestInGenerations = []
    for i in range(population_size):
        individual = generate_valid_solution(len(cities))
        population.append(individual)
    start_time = time.time()
    for i in range(num_generations):
        # calculez fitness si sortez
        fitnesses = [(fitness(solution, cities), solution) for solution in population]
        fitnesses.sort(reverse=True)
        # pastrez elita generatiei anterioare
        elite_count = int(population_size * elite_percentage)
        new_population = [fitnesses[j][1] for j in range(elite_count)]
        # adaug noi indivizi prin incrucisare, mutatie
        for j in range(elite_count, population_size):
            parent1 = selection(population, fitnesses, tournament_size)
            parent2 = selection(population, fitnesses, tournament_size)
            if random.random() < crossover_probability:
                child1, child2 = pmx_crossover(parent1, parent2)
                new_population.append(mutate_swap(child1, mutation_probability))
                new_population.append(mutate_swap(child2, mutation_probability))
            else:
                new_population.append(mutate_swap(parent1, mutation_probability))
                new_population.append(mutate_swap(parent2, mutation_probability))
        new_population = [solution for solution in new_population if isValid(solution)]
        random.shuffle(new_population)
        # pastrez dimensiunea populatiei, pastrz exemplarele bune
        new_population = sorted(new_population, key=lambda x: fitness(x, cities), reverse=True)[:population_size]
        population = new_population
        # determin best solution din generatie
        fitnesses = [(fitness(solution, cities), solution) for solution in population]
        fitnesses.sort(reverse=True)
        bestInGenerations.append(fitnesses[0][1])
        # print("Generation:", i+1, "Best solution:", fitnesses[0][1], "Fitness:", 1/fitnesses[0][0])

    end_time = time.time()
    # determin best solution din toate generatiile
    fitnesses = [(fitness(solution, cities), solution) for solution in bestInGenerations]
    fitnesses.sort(reverse=True)
    best_solution = fitnesses[0][1]
    best_fitness = fitnesses[0][0]
    return best_solution, 1/best_fitness, end_time-start_time

#---------------------
def evolutionary_algorithm_elite1b(population_size, num_generations, crossover_probability, mutation_probability, tournament_size, elite_percentage):
    population = []
    bestInGenerations = []
    for i in range(population_size):
        individual = generate_valid_solution(len(cities))
        population.append(individual)
    start_time = time.time()
    for i in range(num_generations):
        # calculez fitness si sortez
        fitnesses = [(fitness(solution, cities), solution) for solution in population]
        fitnesses.sort(reverse=True)
        # pastrez elita generatiei anterioare
        elite_count = int(population_size * elite_percentage)
        new_population = [fitnesses[j][1] for j in range(elite_count)]
        #-------------------------------------------------

        # adaug noi copii prin incrucisare, mutatie
        children = []
        for j in range(elite_count, population_size):
            parent1 = selection(population, fitnesses, tournament_size)
            parent2 = selection(population, fitnesses, tournament_size)
            if random.random() < crossover_probability:
                child1, child2 = pmx_crossover(parent1, parent2)
                children.append(mutate_swap(child1, mutation_probability))
                children.append(mutate_swap(child2, mutation_probability))
            else:
                children.append(mutate_swap(parent1, mutation_probability))
                children.append(mutate_swap(parent2, mutation_probability))

        # selectam copiii si ii adaugam la noua populatie
        selected_children = selection_children(children, [(fitness(solution, cities), solution) for solution in children],
                                      tournament_size)
        # new_population = [fitnesses[j][1] for j in range(elite_count)]
        new_population.extend(selected_children)
        new_population = [solution for solution in new_population if isValid(solution)]
        random.shuffle(new_population)

        # pastram dimensiunea populatiei si exemplarele bune
        new_population = sorted(new_population, key=lambda x: fitness(x, cities), reverse=True)[:population_size]
        population = new_population

        # determin best solution din generatie
        fitnesses = [(fitness(solution, cities), solution) for solution in population]
        fitnesses.sort(reverse=True)
        bestInGenerations.append(fitnesses[0][1])
        # print("Generation:", i+1, "Best solution:", fitnesses[0][1], "Fitness:", 1/fitnesses[0][0])

    end_time = time.time()
    # determin best solution din toate generatiile
    fitnesses = [(fitness(solution, cities), solution) for solution in bestInGenerations]
    fitnesses.sort(reverse=True)
    best_solution = fitnesses[0][1]
    best_fitness = fitnesses[0][0]
    return best_solution, 1/best_fitness, end_time-start_time

#--------------------
# AE ordered_crossorder, mutate_swap
def evolutionary_algorithm_elite2(population_size, num_generations, crossover_probability, mutation_probability, tournament_size, elite_percentage):
    population = []
    bestInGenerations = []
    for i in range(population_size):
        individual = generate_valid_solution(len(cities))
        population.append(individual)
    start_time = time.time()
    for i in range(num_generations):
        # calculez fitness si sortez
        fitnesses = [(fitness(solution, cities), solution) for solution in population]
        fitnesses.sort(reverse=True)
        # pastrez elita generatiei anterioare
        elite_count = int(population_size * elite_percentage)
        new_population = [fitnesses[j][1] for j in range(elite_count)]
        # adaug noi copii prin incrucisare, mutatie
        children = []
        for j in range(elite_count, population_size):
            parent1 = selection(population, fitnesses, tournament_size)
            parent2 = selection(population, fitnesses, tournament_size)
            if random.random() < crossover_probability:
                child1, child2 = pmx_crossover(parent1, parent2)
                children.append(mutate_swap(child1, mutation_probability))
                children.append(mutate_swap(child2, mutation_probability))
            else:
                children.append(mutate_swap(parent1, mutation_probability))
                children.append(mutate_swap(parent2, mutation_probability))

        # selectam copiii si ii adaugam la noua populatie
        selected_children = selection_children(children,
                                               [(fitness(solution, cities), solution) for solution in children],
                                               tournament_size)
        # new_population = [fitnesses[j][1] for j in range(elite_count)]
        new_population.extend(selected_children)
        new_population = [solution for solution in new_population if isValid(solution)]
        random.shuffle(new_population)

        # pastram dimensiunea populatiei si exemplarele bune
        new_population = sorted(new_population, key=lambda x: fitness(x, cities), reverse=True)[:population_size]
        population = new_population

        # determin best solution din generatie
        fitnesses = [(fitness(solution, cities), solution) for solution in population]
        fitnesses.sort(reverse=True)
        bestInGenerations.append(fitnesses[0][1])
        # print("Generation:", i+1, "Best solution:", fitnesses[0][1], "Fitness:", 1/fitnesses[0][0])
    end_time = time.time()
    # determin best solution
    fitnesses = [(fitness(solution, cities), solution) for solution in bestInGenerations]
    # f=[]
    # generations = list(range(num_generations))
    # for x in fitnesses:
    #     f.append(x[0])
    # # for sol in fitness_values:
    # #     print(sol)
    # plt.plot(generations, f)
    # # plt.bar(iteratii, fitness_values)
    # plt.title('Fitness Values by Generations')
    # plt.xlabel('Generations')
    # plt.ylabel('Fitness')
    # plt.show()

    fitnesses.sort(reverse=True)
    best_solution = fitnesses[0][1]
    best_fitness = fitnesses[0][0]
    return best_solution, 1/best_fitness, end_time-start_time

def selection_children(children, fitnesses, tournament_size):
    selected = []
    while len(selected) < len(children):
        tournament = random.sample(fitnesses, tournament_size)
        winner = max(tournament, key=lambda x: x[0])[1]
        selected.append(winner)
    return selected

def proportional_selection(solutions, fitnesses, tournament_size):
    selected_solutions = []
    selected_fitnesses = []
    for i in range(tournament_size):
        index = random.randint(0, len(solutions) - 1)
        selected_solutions.append(solutions[index])
        selected_fitnesses.append(fitnesses[index])
    return selected_solutions[proportional_index(selected_fitnesses)]

def proportional_index(fitnesses):
    total_fitness = sum(f[0] for f in fitnesses)
    r = random.uniform(0, total_fitness)
    partial_sum = 0
    for i in range(len(fitnesses)):
        partial_sum += fitnesses[i][0]
        if partial_sum >= r:
            return i



# parsez fisierul intr-o lista de coordonate ale oraselor
def parse_file_tsp(filename):
    with open(filename, 'r') as file:
        linii = file.readlines()
    cities = []
    for i in range(6, len(linii) - 1):
        oras = linii[i].strip().split()
        coord_x = int(oras[1])
        coord_y = int(oras[2])
        cities.append((coord_x, coord_y))

    return cities

def isValid(solution):
    return len(set(solution)) == len(solution)

def result(file, population_size, num_generations, crossover_probability,
                                       mutation_probability, tournament_size, elite_percentage):
    with open(file, 'w') as file:
        solutions_tsp1 = []
        solutions_tsp2 = []
        print(f'Pentru problema TSP avem  {len(cities)} orase\n', file=file)
        for i in range(1, 11):
            print(f' Iteratia {i}: \n', file=file)
            solution1 = evolutionary_algorithm_elite1b(population_size, num_generations, crossover_probability, mutation_probability, tournament_size, elite_percentage)
            solutions_tsp1.append(solution1)
            print(
                f'Pentru problema TSP1 avem solutia: {solution1[0]}\n cu fitness: {solution1[1]} \n timp de executie {solution1[2]}\n',
                file=file)
            solution2 = evolutionary_algorithm_elite2(population_size, num_generations, crossover_probability, mutation_probability, tournament_size, elite_percentage)

            solutions_tsp2.append(solution2)
            print(
                f'Pentru problema TSP2 avem solutia: {solution2[0]}\n cu fitness:  {solution2[1]}\n timp de executie {solution2[2]}\n',
                file=file)
        # iteratii = list(range(1, 11))
        # fitness_values1 = []
        # fitness_values2 = []
        # for sol in solutions_tsp2:
        #     fitness_values2.append(sol[1])
        # for sol in solutions_tsp1:
        #     fitness_values1.append(sol[1])
        # # for sol in fitness_values:
        # #     print(sol)
        # plt.plot(iteratii, fitness_values1)
        # plt.plot(iteratii, fitness_values2)
        # # plt.bar(iteratii, fitness_values)
        # plt.title('Fitness Values by Iteration')
        # plt.xlabel('Iteration')
        # plt.ylabel('Fitness')
        # plt.show()


        print(f'------Sumar------\n\n'
              f'-- TSP -- (minimizare) --    Avem {len(cities)} orase\n'
              f'--population_size {population_size}, num_generations {num_generations}, crossover_probability {crossover_probability}, mutation_probability {mutation_probability}, tournament_size {tournament_size}, elite_percentage {elite_percentage} \n'
              f' - AE pentru problema TSP: (var 1 - incrucisare ordonata, mutatie swap)  \n', file = file)
        print_sumar(file, solutions_tsp1, minim(solutions_tsp1), maxim(solutions_tsp1))
        print(f' - AE pentru problema TSP2: (var 2 - incrucisare ciclica, mutatie reversed) \n', file=file)
        print_sumar(file, solutions_tsp2, minim(solutions_tsp2), maxim(solutions_tsp2))

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
# tsp

tsp_file = "kroB10_tsp.txt"
cities = parse_file_tsp(tsp_file)
# population_size = int(input('Marime populatie:  '))
# num_generations = int(input('Nr generatii:  '))
# crossover_probability = int(input('Probabilitate de incrucisare <1:  '))
# mutation_probability = int(input('Probabilitate de mutatie <1:  '))
# elite_percentage = int(input('Procentul elitei <1:  '))
# rulam algoritmul evolutiv elitist cu parametrii specificati


population_size=100
num_generations=50
crossover_probability=0.5
mutation_probability=0.5
tournament_size = 2
elite_percentage = 0.1
result_file = "tsp-test.txt"
result(result_file, population_size, num_generations, crossover_probability,
                                       mutation_probability, tournament_size, elite_percentage)
