import math
import random
import time


def parse_file_rucsac(filename):
    greutati = []
    valori = []
    elemente = []
    with open(filename) as file:
        n = int(file.readline().strip())
        linii = file.readlines()
        capacitate = int(linii[-1].strip())  # iau ultima linie
        linii = linii[:-1]  # elimin ultima linie
        for linie in linii[0:]:
            linie = linie.split()  # 0 -index obiect, 1-valoare, 2-greutate
            valori.append(int(linie[1]))
            greutati.append(int(linie[2]))
            elemente.append((int(linie[1]), int(linie[2])))
    return n, capacitate, greutati, valori


def parse_file_tsp(filename):
    with open(filename, 'r') as file:
        linii = file.readlines()
    coordonate_orase = []
    for i in range(6, len(linii) - 1):
        oras = linii[i].strip().split()
        coord_x = int(oras[1])
        coord_y = int(oras[2])
        coordonate_orase.append((coord_x, coord_y))

    return coordonate_orase


def afisare_date_fisier_tsp(nume_fisier):
    lista_coordonate = parse_file_tsp(nume_fisier)
    print('Numarul de orase este: ' + str(len(lista_coordonate)))
    print('Lista de coordonate este: ' + str(lista_coordonate))


def genereaza_solutie():
    x = [random.choice(range(2)) for _ in range(n)]
    return x


def solutie_valida(x):
    greutate = 0
    for y in range(len(x)):
        if (x[y] == 1):
            greutate = greutate + greutati[y]
    if greutate > capacitate:
        return False
    return True


def calcul_fitness(x):
    calitate = 0
    for y in range(len(x)):
        calitate = calitate + x[y] * valori[y]
    return calitate


def genereaza_vecin(x):
    vecin = x[:]
    index = random.randint(0, n - 1)
    # print(index)
    if vecin[index] == 1:
        vecin[index] = 0
    else:
        vecin[index] = 1
    return vecin


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


def tabu_search(k, tabu_size):
    start_time = time.perf_counter()

    x = genereaza_solutie()
    while not solutie_valida(x):  # pornim cu o solutie valida
        x = genereaza_solutie()
    best = x
    tabu_list = []

    for i in range(k):
        best_vecin = None
        best_fitness_vecin = -1

        for j in range(n):  # generam vecinii
            vecin = genereaza_vecin(x)
            while not solutie_valida(vecin):
                vecin = genereaza_vecin(x)

            if (vecin not in tabu_list) and (calcul_fitness(vecin) > best_fitness_vecin):
                best_vecin = vecin
                best_fitness_vecin = calcul_fitness(vecin)
        tabu_list.append(x)
        if len(tabu_list) > tabu_size:  # elimin primul element daca se depaseste lungimea data
            tabu_list.pop(0)

        if best_fitness_vecin > calcul_fitness(best):
            best = best_vecin
        x = best_vecin
    end_time = time.perf_counter()
    run_time = end_time - start_time

    return best, calcul_fitness(best), run_time


################################################

def calculeaza_distanta(oras1, oras2):
    x1, y1 = oras1
    x2, y2 = oras2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculeaza_distanta_ruta(orase, ruta):
    distance = 0
    for i in range(len(ruta)):
        j = (i + 1) % len(ruta)
        oras1 = orase[ruta[i]]
        oras2 = orase[ruta[j]]
        distance += calculeaza_distanta(oras1, oras2)
    return distance

# generează o listă cu toate posibilele soluții vecine pentru o rută dată.
# -generarea unei noi rute prin inversarea ordinii orașelor între doua orașe.
# -returneaza o lista de soluții vecine este returnată.

def vecinatate(ruta):
    vecini = []
    for i in range(1, len(ruta) - 2):
        for j in range(i + 2, len(ruta)):
            ruta_noua = ruta[:]
            ruta_noua[i:j] = reversed(ruta_noua[i:j])
            vecini.append(ruta_noua)
    return vecini


def vecinatate_best(orase, tabu_list, ruta_curenta):
    best_distanta = float('inf')
    best_ruta = None
    for ruta in vecinatate(ruta_curenta):
        if ruta not in tabu_list:
            distanta = calculeaza_distanta_ruta(orase, ruta)
            if distanta < best_distanta:
                best_distanta = distanta
                best_ruta = ruta
    return best_ruta, best_distanta


def tsp_tabu_search(orase, nr_max_iteratii, tabu_size):
    start_time = time.perf_counter()
    ruta_curenta = list(range(len(orase)))
    random.shuffle(ruta_curenta)  # generez o lista de orase pe care o amestec => solutie random
    best_ruta = ruta_curenta[:]
    tabu_list = []
    best_distanta = calculeaza_distanta_ruta(orase, best_ruta)

    for i in range(nr_max_iteratii):
        distanta_curenta = calculeaza_distanta_ruta(orase, ruta_curenta)
        if distanta_curenta < best_distanta:
            best_distanta = distanta_curenta
            best_ruta = ruta_curenta[:]

        vecini = vecinatate(ruta_curenta)
        vecini_sortati = sorted(vecini, key=lambda x: calculeaza_distanta_ruta(orase, x))
        for ruta in vecini_sortati:
            if ruta not in tabu_list:
                ruta_curenta = ruta
                break

        tabu_list.append(ruta_curenta)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)
    end_time = time.perf_counter()
    run_time = end_time - start_time

    return best_ruta, best_distanta, run_time


################################################
def rezultat(k, f1):
    with open(f1, 'w') as file:
        solutii_rucsac = []
        solutii_tsp = []
        print(f'Pentru problema rucsacului avem n = {n}\n'
              f'Capacitate rucsac = {capacitate}\n', file=file)
        print(f'Pentru problema TSP avem  {len(coordonate_orase)} orase\n', file=file)
        for i in range(1, 11):
            print(f' Iteratia {i}: \n', file=file)

            solutii_rucsac.append(tabu_search(k, 20))
            print(
                f'Pentru problema rucsacului avem solutia: {solutii_rucsac[i - 1][0]}\n cu fitness: {solutii_rucsac[i - 1][1]}\n timp de executie {solutii_rucsac[i - 1][2]}\n',
                file=file)
            solutii_tsp.append(tsp_tabu_search(coordonate_orase, k, 20))
            print(
                f'Pentru problema TSP avem solutia: {solutii_tsp[i - 1][0]}\n cu distanta minima:  {solutii_tsp[i - 1][1]}\n timp de executie {solutii_tsp[i - 1][2]}\n',
                file=file)

        print(f'------Sumar------\n\n'
              f'-- Avem {n} elemente in rucsac, {len(coordonate_orase)}  orase si k = {k} \n'
              f' - tabu search pentru problema rucsacului: (maximizare)  \n', file = file)
        print_sumar(k, file, solutii_rucsac, maxim(solutii_rucsac), minim(solutii_rucsac))
        print(f' - tabu search pentru TSP: (minimizare) \n', file=file)
        print_sumar(k, file, solutii_tsp, minim(solutii_tsp), maxim(solutii_tsp))


def print_sumar(k, file, solutii, best, worst):
    print(f'cea mai buna solutie a fost: {solutii[best[1]][0]},\n calitate = {best[0]}\n', file=file)
    print(f'cea mai slaba solutie a fost: {solutii[worst[1]][0]},\n calitate = {worst[0]}\n', file=file)
    total = 0
    timpi = 0
    for sol in solutii:
        total += sol[1]
        timpi += sol[2]
    print(f'calitatea medie:  {total / len(solutii)}\n', file=file)
    print(f'timp mediu de executie: {timpi / len(solutii)}\n', file=file)


################################################

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


if __name__ == '__main__':
    print_hi('PyCharm')

    # n = 10
    # capacitate = 300
    # greutati = [10, 12, 28, 14, 30, 26, 12, 8, 24, 10]
    # valori = [45, 23, 87, 10, 35, 20, 32, 7, 20, 30]

    # coordonate_orase = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10),
    #                     (11, 11),
    #                     (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19)]

    file1 = 'rucsac-20.txt'
    file2 = 'kroB100_tsp.txt'
    f1 = 'test.txt'
    k = int(input('k = '))
    n, capacitate, greutati, valori = parse_file_rucsac(file1)
    coordonate_orase = parse_file_tsp(file2)
    rezultat(k, f1)
