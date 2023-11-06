import numpy as np
import random
import time
import matplotlib.pyplot as plt

#norma euclidiana a unui vector
def distance(city1, city2):
    return np.linalg.norm(np.array(city1) - np.array(city2))

def create_distance_matrix(cities):
    n = len(cities)
    dist_matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i][j] = distance(cities[i], cities[j])
    return dist_matrix


def construct_ant_tour(num_cities, dist_matrix, alpha, beta, tau):
    visited = set()
    current_city = random.randint(0, num_cities - 1)
    visited.add(current_city)
    route = [current_city]
    cost = 0

    for step in range(num_cities - 1):
        probabilities = np.zeros(num_cities)
        denominator = 0

        # probabilitatea de a alege urmatorul oras, dupa formula din seminar
        for city in range(num_cities):
            if city not in visited:
                numerator = tau[current_city][city] ** alpha * (1.0 / dist_matrix[current_city][city]) ** beta
                probabilities[city] = numerator
                denominator += numerator

        probabilities /= denominator
        # Funcția np.random.choice() selectează apoi în mod aleatoriu un oraș nevizitat pe baza acestor probabilități.
        next_city = np.random.choice(range(num_cities), p=probabilities)
        visited.add(next_city)
        route.append(next_city)
        cost += dist_matrix[current_city][next_city]
        current_city = next_city

    cost += dist_matrix[route[-1]][route[0]]  # adaugam ultima muchie pt a completa ciclul

    return route, cost

def validate_solution(route, num_cities):
    visited = set()
    for city in route:
        if city in visited:
            return False
        visited.add(city)
    if len(visited) != num_cities:
        return False
    if route[0] != route[-1]:
        return False
    return True


def local_search(best_route, best_cost, dist_matrix):
    num_cities = len(best_route)

    for i in range(num_cities - 1):
        j = i + 1
        while j < num_cities:
            new_route = best_route[:]
            new_route[i:j + 1] = reversed(new_route[i:j + 1])
            new_cost = calculate_cost(new_route, dist_matrix)

            if validate_solution(new_route, num_cities) and new_cost < best_cost:
                best_route = new_route
                best_cost = new_cost
            j += 1

    return best_route, best_cost

def calculate_cost(route, dist_matrix):
    cost = 0
    for i in range(len(route)-1):
        cost += dist_matrix[route[i]][route[i+1]]
    cost += dist_matrix[route[-1]][route[0]] # inchidem bucla
    return cost

#var1 - doar pe baza de feromoni
def ant_system_tsp(cities, num_ants, alpha, beta, evaporation_rate, max_iterations):
    num_cities = len(cities)
    dist_matrix = create_distance_matrix(cities)
    tau = np.ones((num_cities, num_cities))  # initialize pheromone matrix with 1's
    best_cost = float('inf')
    best_route = []

    start_time = time.time()
    for iteration in range(max_iterations):
        routes = []
        costs = []

        for ant in range(num_ants):
            route, cost = construct_ant_tour(num_cities, dist_matrix, alpha, beta, tau)
            routes.append(route)
            costs.append(cost)

        if min(costs) < best_cost:
            best_cost = min(costs)
            best_route = routes[costs.index(best_cost)]

        delta_tau = np.zeros((num_cities, num_cities))

        for r in range(num_ants):
            for i in range(num_cities - 1):
                delta_tau[routes[r][i]][routes[r][i + 1]] += 1.0 / costs[r]

            delta_tau[routes[r][-1]][routes[r][0]] += 1.0 / costs[r]

        tau = (1 - evaporation_rate) * tau + delta_tau  # pheromon trail update
    end_time = time.time()

    return  best_route, best_cost, end_time - start_time

# var2 +cautare locala, exploram vecinatatea
def ant_system_tsp1(cities, num_ants, alpha, beta, evaporation_rate, max_iterations):
    num_cities = len(cities)
    dist_matrix = create_distance_matrix(cities)
    tau = np.ones((num_cities, num_cities))  # initializam matricea de feromoni
    best_cost = float('inf')
    best_route = []

    start_time = time.time()
    for iteration in range(max_iterations):
        routes = []
        costs = []

        #
        for ant in range(num_ants):
            route, cost = construct_ant_tour(num_cities, dist_matrix, alpha, beta, tau)

            routes.append(route)
            costs.append(cost)

        if min(costs) < best_cost:
            best_cost = min(costs)
            best_route = routes[costs.index(best_cost)]

        delta_tau = np.zeros((num_cities, num_cities))

        for r in range(num_ants):
            for i in range(num_cities - 1):
                delta_tau[routes[r][i]][routes[r][i + 1]] += 1.0 / costs[r]

            delta_tau[routes[r][-1]][routes[r][0]] += 1.0 / costs[r]

        tau = (1 - evaporation_rate) * tau + delta_tau  # pheromon trail update

    best_route, best_cost = local_search(best_route, best_cost, dist_matrix)
    end_time = time.time()

    return  best_route, best_cost, end_time - start_time



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

def result(file, cities, num_ants, alpha, beta, evaporation_rate, max_iterations):
    with open(file, 'w') as file:
        solutions_tsp1 = []
        solutions_tsp2 = []
        print(f'Pentru problema ACO TSP avem  {len(cities)} orase\n', file=file)
        for i in range(1, 11):
            print(f' Iteratia {i}: \n', file=file)
            solution1 = ant_system_tsp(cities, num_ants, alpha, beta, evaporation_rate, max_iterations)
            solutions_tsp1.append(solution1)
            print(
                f'Pentru problema TSP1 avem solutia: {solution1[0]}\n cu fitness: {solution1[1]} \n timp de executie {solution1[2]}\n',
                file=file)
            solution2 = ant_system_tsp1(cities, num_ants, alpha, beta, evaporation_rate, max_iterations)
            solutions_tsp2.append(solution2)
            print(
                f'Pentru problema TSP2 avem solutia: {solution2[0]}\n cu fitness:  {solution2[1]}\n timp de executie {solution2[2]}\n',
                file=file)
        iteratii = list(range(1, 11))
        fitness_values1 = []
        fitness_values2 = []
        for sol in solutions_tsp2:
            fitness_values2.append(sol[1])
        for sol in solutions_tsp1:
            fitness_values1.append(sol[1])
        # for sol in fitness_values:
        #     print(sol)
        plt.plot(iteratii, fitness_values1)
        plt.plot(iteratii, fitness_values2)
        # plt.bar(iteratii, fitness_values)
        plt.title('Fitness Values by Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.show()


        print(f'------Sumar------\n\n'
              f'-- TSP -- (minimizare) --    Avem {len(cities)} orase\n'
              f' - ACO pentru problema TSP1: (var 1 - pe baza de feromoni)  \n', file = file)
        print_sumar(file, solutions_tsp1, minim(solutions_tsp1), maxim(solutions_tsp1))
        print(f' - ACO pentru problema TSP2: (var 2 - feromoni + cautare locala) \n', file=file)
        print_sumar(file, solutions_tsp2, minim(solutions_tsp2), maxim(solutions_tsp2))

def print_sumar(file, solutii, best, worst):
    print(f'cea mai buna solutie a fost: {solutii[best[1]][0]},\n calitate = {best[0]}\n', file=file)
    # print(f'cea mai slaba solutie a fost: {solutii[worst[1]][0]},\n calitate = {worst[0]}\n', file=file)
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



###############
# cities = [(0,0), (1,5), (2,3), (5,4), (8,0), (6,3)]
tsp_file = "kroB100_tsp.txt"
cities = parse_file_tsp(tsp_file)
num_ants=100
alpha=1
beta=1
evaporation_rate=0.5
max_iterations=100

result_file = "aco_tsp-100-100-100.txt"
result(result_file, cities, num_ants, alpha, beta, evaporation_rate, max_iterations)

