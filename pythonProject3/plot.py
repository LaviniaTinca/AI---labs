import matplotlib.pyplot as plt


# iteratii = list(range(1, 11))
# fitness_values1 = []
# fitness_values2 = []
# for sol in solutions_tsp2:
#     fitness_values2.append(sol[1])
# for sol in solutions_tsp1:
#     fitness_values1.append(sol[1])
# # for sol in fitness_values:
# #     print(sol)
# best = [781, 787, 787]
# media = [755, 776, 781]
# generatii = [100, 500, 1000]
#
# plt.plot(generatii, best)
# plt.plot(generatii, media)
# # plt.bar(iteratii, fitness_values)
# plt.title('Knapsack 20 AE2, BEST vs AVERAGE')
# plt.xlabel('Generatii')
# plt.ylabel('Fitness')
# plt.show()

best = [8970, 8970, 9042]
media = [9393, 9283, 9286]
generatii = [100, 500, 1000]

plt.plot(generatii, best)
plt.plot(generatii, media)
# plt.bar(iteratii, fitness_values)
plt.title('TSP 10 AE2, BEST vs AVERAGE')
plt.xlabel('Generatii')
plt.ylabel('Fitness')
plt.show()