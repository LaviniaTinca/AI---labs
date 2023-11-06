import random
import time
import math
import matplotlib.pyplot as plt


def generate_valid_solution(dimensions):
    return [random.uniform(-5.12, 5.12) for _ in range(dimensions)]

#rastrigin function
def fitness(solution):
    n = len(solution)
    fitness_value = 10 * n
    for i in range(n):
        fitness_value += solution[i]**2 - 10 * math.cos(2 * math.pi * solution[i])
    return fitness_value

def selection(population, fitnesses, tournament_size):
    tournament = random.sample(range(len(population)), tournament_size)
    tournament_fitnesses = [fitnesses[i][0] for i in tournament]
    best_index = tournament_fitnesses.index(min(tournament_fitnesses))
    return population[tournament[best_index]]

def arithmetic_crossover(parent1, parent2):
    alpha = random.uniform(0, 1)
    child1 = []
    child2 = []
    for gene1, gene2 in zip(parent1, parent2):
        child_gene1 = alpha * gene1 + (1 - alpha) * gene2
        child_gene2 = (1 - alpha) * gene1 + alpha * gene2
        child1.append(child_gene1)
        child2.append(child_gene2)

    return child1, child2

def uniform_crossover(parent1, parent2):
    child1 = []
    child2 = []
    for gene1, gene2 in zip(parent1, parent2):
        if random.random() < 0.5:
            child1.append(gene1)
            child2.append(gene2)
        else:
            child1.append(gene2)
            child2.append(gene1)
    return child1, child2


def polynomial_mutation(solution, mutation_probability, distribution_index):
    mutated_solution = solution[:]
    size = len(solution)

    for i in range(size):
        if random.random() < mutation_probability:
            gene = mutated_solution[i]
            delta = min(gene - (-5.12), 5.12 - gene)
            u = random.random()
            if u <= 0.5:
                delta_q = (2.0 * u) ** (1.0 / (distribution_index + 1)) - 1.0
            else:
                delta_q = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (distribution_index + 1))
            mutated_gene = gene + delta * delta_q
            mutated_solution[i] = min(max(mutated_gene, -5.12), 5.12)

    return mutated_solution


def mutate_gaussian(solution, mutation_probability, mutation_strength):
    mutated_solution = solution[:]
    size = len(solution)

    for i in range(size):
        if random.random() < mutation_probability:
            mutated_solution[i] += random.gauss(0, mutation_strength)

    return mutated_solution


def selection_children(children, children_fitnesses, tournament_size):
    # Perform tournament selection for the children population
    selected_children = []

    while len(selected_children) < len(children):
        tournament = random.sample(range(len(children)), tournament_size)
        tournament_fitnesses = [children_fitnesses[i][0] for i in tournament]
        best_index = tournament_fitnesses.index(max(tournament_fitnesses))
        selected_children.append(children[tournament[best_index]])

    return selected_children

def isValid(solution):
    for value in solution:
        if value < -5.12 or value > 5.12:
            return False
    return True

# mutatie gaussiana, arithmetic_crossover
def evolutionary_algorithm_elite1(population_size, num_generations, crossover_probability, mutation_probability,
                                  tournament_size, elite_percentage, dimensions):
    global best_solution, best_fitness
    population = []
    bestInGenerations = []
    for i in range(population_size):
        individual = generate_valid_solution(dimensions)
        population.append(individual)
    start_time = time.time()
    for i in range(num_generations):
        fitnesses = [(fitness(solution), solution) for solution in population]
        fitnesses.sort(reverse=True)
        elite_count = int(population_size * elite_percentage)
        new_population = [fitnesses[j][1] for j in range(elite_count)]
        children = []
        for j in range(elite_count, population_size):
            parent1 = selection(population, fitnesses, tournament_size)
            parent2 = selection(population, fitnesses, tournament_size)
            if random.random() < crossover_probability:
                child1, child2 = arithmetic_crossover(parent1, parent2)
                children.append(mutate_gaussian(child1, mutation_probability, mutation_strength=0.2))
                children.append(mutate_gaussian(child2, mutation_probability, mutation_strength=0.2))
            else:
                children.append(mutate_gaussian(parent1, mutation_probability, mutation_strength=0.2))
                children.append(mutate_gaussian(parent2, mutation_probability, mutation_strength=0.2))

        selected_children = selection_children(children, [(fitness(solution), solution) for solution in children],
                                               tournament_size)
        new_population.extend(selected_children)
        new_population = [solution for solution in new_population if isValid(solution)]
        new_population = sorted(new_population, key=lambda x: fitness(x), reverse=True)[:population_size]
        random.shuffle(new_population)
        population = new_population

        # Determine the best solution in the current generation
        best_solution = fitnesses[0][1]
        best_fitness = fitnesses[0][0]
        bestInGenerations.append((1/best_fitness, best_solution))

    end_time = time.time()
    return best_solution, 1 / best_fitness, end_time - start_time, bestInGenerations


# uniform_crossover, polinomial mutatian
def evolutionary_algorithm_elite2(population_size, num_generations, crossover_probability, mutation_probability,
                                  tournament_size, elite_percentage, dimensions):
    population = []
    bestInGenerations = []
    for i in range(population_size):
        individual = generate_valid_solution(dimensions)
        population.append(individual)
    start_time = time.time()
    for i in range(num_generations):
        fitnesses = [(fitness(solution), solution) for solution in population]
        fitnesses.sort(reverse=True)
        elite_count = int(population_size * elite_percentage)
        new_population = [fitnesses[j][1] for j in range(elite_count)]
        children = []
        for j in range(elite_count, population_size):
            parent1 = selection(population, fitnesses, tournament_size)
            parent2 = selection(population, fitnesses, tournament_size)
            if random.random() < crossover_probability:
                # child1, child2 = arithmetic_crossover(parent1, parent2)
                child1, child2 = uniform_crossover(parent1, parent2)

                # children.append(mutate_gaussian(child1, mutation_probability, mutation_strength=0.2))
                children.append(polynomial_mutation(child1, mutation_probability, distribution_index=10))
                children.append(polynomial_mutation(child2, mutation_probability, distribution_index=10))
                # children.append(mutate_gaussian(child2, mutation_probability, mutation_strength=0.2))
            else:
                # children.append(mutate_gaussian(parent1, mutation_probability, mutation_strength=0.2))
                # children.append(mutate_gaussian(parent2, mutation_probability, mutation_strength=0.2))
                children.append(polynomial_mutation(parent1, mutation_probability, distribution_index=10))
                children.append(polynomial_mutation(parent2, mutation_probability, distribution_index=10))
        selected_children = selection_children(children, [(fitness(solution), solution) for solution in children],
                                               tournament_size)
        new_population.extend(selected_children)
        new_population = [solution for solution in new_population if isValid(solution)]
        new_population = sorted(new_population, key=lambda x: fitness(x), reverse=True)[:population_size]
        random.shuffle(new_population)
        population = new_population

        # Determine the best solution in the current generation
        best_solution = fitnesses[0][1]
        best_fitness = fitnesses[0][0]
        bestInGenerations.append((1/best_fitness, best_solution))

    end_time = time.time()

    # f=[]
    # generations = list(range(num_generations))
    # for x in bestInGenerations:
    #     f.append(x[0])

    # plt.plot(generations, f)
    # plt.bar(generations, f)
    # plt.title('Fitness Values by Generations')
    # plt.xlabel('Generations')
    # plt.ylabel('Fitness')
    # plt.show()
    return best_solution, 1 / best_fitness, end_time - start_time, bestInGenerations

def result(file, population_size, num_generations, crossover_probability,
                                       mutation_probability, tournament_size, elite_percentage, dimensions):
    with open(file, 'w') as file:
        solutions_1 = []
        solutions_2 = []
        print(f'Pentru functia Rastrigin avem  {dimensions} - spatiul de cautare\n', file=file)
        for i in range(1, 11):
            print(f' Iteratia {i}: \n', file=file)
            solution1 = evolutionary_algorithm_elite1(population_size, num_generations, crossover_probability, mutation_probability, tournament_size, elite_percentage, dimensions)
            solutions_1.append(solution1)
            print(
                f'Pentru functia Rastrigin 1 avem solutia: {solution1[0]}\n cu fitness: {solution1[1]} \n timp de executie {solution1[2]}\n',
                file=file)
            solution2 = evolutionary_algorithm_elite2(population_size, num_generations, crossover_probability, mutation_probability, tournament_size, elite_percentage, dimensions)

            solutions_2.append(solution2)
            print(
                f'Pentru problema Rastrigin 2 avem solutia: {solution2[0]}\n cu fitness:  {solution2[1]}\n timp de executie {solution2[2]}\n',
                file=file)
        # iteratii = list(range(1, 11))
        # fitness_values1 = []
        # fitness_values2 = []
        # for sol in solutions_2:
        #     fitness_values2.append(sol[1])
        # for sol in solutions_1:
        #     fitness_values1.append(sol[1])
        # # for sol in fitness_values:
        # #     print(sol)
        # plt.rcParams["interactive.backend"] = "TkAgg"
        # plt.plot(iteratii, fitness_values1)
        # plt.plot(iteratii, fitness_values2)
        # # plt.bar(iteratii, fitness_values)
        # plt.title('Fitness Values by Iteration')
        # plt.xlabel('Iteration')
        # plt.ylabel('Fitness')
        # plt.show()


        print(f'------Sumar------\n\n'
              f'-- Rastrigin -- (minimizare) --    Avem {dimensions} -spatiul de cautare\n'
              f'--population_size {population_size}, num_generations {num_generations}, crossover_probability {crossover_probability}, mutation_probability {mutation_probability}, tournament_size {tournament_size}, elite_percentage {elite_percentage} \n'
              f' - AE pentru problema Rastrigin: (var 1 - incrucisare aritmetica, mutatie gaussiana)  \n', file = file)
        print_sumar(file, solutions_1, minim(solutions_1), maxim(solutions_1))
        print(f' - AE pentru problema Rastrigin: (var 2 - incrucisare uniforma, mutatie polinomiala) \n', file=file)
        print_sumar(file, solutions_2, minim(solutions_2), maxim(solutions_2))

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


######################################################################################


# Example usage
population_size = 500
num_generations = 500
crossover_probability = 0.5
mutation_probability = 0.5
tournament_size = 2
elite_percentage = 0.1
# Number of dimensions for Rastrigin function
dimensions = 100 # spatiul de cautare ~ 20, 200 rucsac sau 10, 100 tsp

result_file = "rastrigin-100-500-500.txt"
result(result_file, population_size, num_generations, crossover_probability,
                                       mutation_probability, tournament_size, elite_percentage, dimensions)


#############
# for i in range(1, 11):
#     best_solution, best_fitness, execution_time, best_in_generations = evolutionary_algorithm_elite2(population_size, num_generations, crossover_probability, mutation_probability, tournament_size, elite_percentage, dimensions)
#
# # # Print the best fitness and solution in each generation
# # for generation, (fitness, solution) in enumerate(best_in_generations):
# #     print(f"Generation {generation+1}: Best Fitness = {fitness}, Best Solution = {solution}")
#
#     # Print the overall best fitness, solution, and execution time
#     print(f"Overall Best Fitness = {best_fitness}, Overall Best Solution = {best_solution}")
#     print(f"Execution Time: {execution_time} seconds")
#     print(f"-----------------------")
