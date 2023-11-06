import random
import time
def parse_file(filename):
    greutati = []
    valori = []
    with open(filename) as file:
        n = int(file.readline().strip())
        linii = file.readlines()
        capacitate = int(linii[-1].strip())  # iau ultima linie
        linii = linii[:-1]  # elimin ultima linie
        for linie in linii[0:]:
            linie = linie.split()  # 0 -index obiect, 1-valoare, 2-greutate
            valori.append(int(linie[1]))
            greutati.append(int(linie[2]))
    return n, capacitate, greutati, valori

def genereaza_solutie():
    x = [random.choice(range(2)) for _ in range(n)]
    return x

def best_solution(fitness):
    best = 0
    index=0
    for i in range(len(fitness)):
        if best < fitness[i]:
            best = fitness[i]
            index = i
    return best, index

def worst_solution(fitness):
    worst, index  = best_solution(fitness)
    for i in range(len(fitness)):
        if worst > fitness[i]:
            worst = fitness[i]
            index = i
    return worst, index

def average(array):
    return sum(array)/len(array)

def solutie_valida(x):
    greutate = 0
    for y in range(len(x)):
        if (x[y]==1):
            greutate = greutate + greutati[y]
    if greutate > capacitate:
        return False
    return True

def calcul_fitness(x):
    calitate = 0
    for y in range(len(x)):
        calitate = calitate + x[y] * valori[y]
    return calitate

def cautare_aleatoare(k, file):
    start_time = time.perf_counter()
    solutii = []
    fitness = []
    for i in range(k):
        calitate = 0
        x = genereaza_solutie()
        print(f'')
        if solutie_valida(x):
            calitate = calcul_fitness(x)
            print(f'{x}, {calitate}\n', file = file)
            solutii.append(x)
            fitness.append(calitate)
    calitate, index_best = best_solution(fitness)
    end_time = time.perf_counter()
    run_time = end_time - start_time
    print(f'Timp de executie {run_time} \n\n', file=file)
    return solutii[index_best], calitate, run_time

def genereaza_vecin(x):
    vecin = x
    index = random.randint(0, n-1)
    if vecin[index] == 1: # flip ultimul bit
        vecin[index] = 0
    else:
        vecin[index] = 1
    return vecin

def rhc(k, file):
    start_time = time.perf_counter()
    x = genereaza_solutie()
    while not solutie_valida(x):
        x = genereaza_solutie()
    calitate = calcul_fitness(x)
    while k > 0:
        vecin = genereaza_vecin(x)
        calitate_vecin = calcul_fitness(vecin)
        if solutie_valida(vecin) and calitate_vecin > calitate :
            x = vecin
            calitate = calitate_vecin
            k = k -1
        else:
            k = k -1
    print(f'{x}, {calitate}\n', file = file)
    end_time = time.perf_counter()
    run_time = end_time - start_time
    print(f'Timp de executie {run_time} \n\n', file=file)
    return  x, calitate, run_time

def rezultat(k, f):
    with open(f, 'w') as file:
        solutii_cautare_aleatoare = []
        fitness_cautare_aleatoare = []
        timpi_ca = []
        solutii_rhc = []
        fitness_rhc = []
        timpi_rhc = []
        print(f'n = {n}\n'
              f'Capacitate rucsac = {capacitate}\n', file = file)
        for i in range(1,11):
            print(f'Iteratia {i} are rezultatele valide: \n', file=file)
            print(f'pentru Cautare Aleatoare:\n', file = file)
            x, calitate, run_time = cautare_aleatoare(k,file)
            solutii_cautare_aleatoare.append(x)
            fitness_cautare_aleatoare.append(calitate)
            timpi_ca.append(run_time)
            print(f'pentru Random Hill Climbing: ', file=file)
            x, calitate, run_time = rhc(k, file)
            solutii_rhc.append(x)
            fitness_rhc.append(calitate)
            timpi_rhc.append(run_time)
        print(f'------Sumar------\n\n'
              f' - pentru cautare aleatoare:  \n', file = file)
        print_sumar(k, file, solutii_cautare_aleatoare, fitness_cautare_aleatoare, timpi_ca)
        print(f' - pentru Random Hill Climbing: \n', file = file)
        print_sumar(k, file, solutii_rhc, fitness_rhc, timpi_rhc)

def print_sumar(k, file, solutii, fitness, timpi):
    print(f'Pentru n = {n}, si k = {k} :', file = file)
    best, index_best = best_solution(fitness)
    worst, index_worst = worst_solution(fitness)
    print(f'cea mai buna solutie a fost: {solutii[index_best]},\n calitate = {best}\n', file=file)
    print(f'cea mai slaba solutie a fost: {solutii[index_worst]},\n calitate = {worst}\n', file=file)
    print(f'calitatea medie: {average(fitness)}\n', file=file)
    print(f'timp mediu de executie: {average(timpi)}\n', file=file)




if __name__ == '__main__':

    # n = 10
    # capacitate = 300
    # greutati = [10, 12, 28, 14, 30, 26, 12, 8, 24, 10]
    # valori = [45, 23, 87, 10, 35, 20, 32, 7, 20, 30]
    k = int(input('k = '))
    # k = 10
    file = 'rucsac-200.txt'
    f1 = 'rezultat.txt'
    n, capacitate, greutati, valori = parse_file(file)
    rezultat(k, f1)