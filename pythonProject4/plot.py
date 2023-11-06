import matplotlib.pyplot as plt

# # Datele de intrare aco var 1
# nr_iteratii = [10, 100, 1000]
# best = [9165.84, 8970.49, 8970.49]
# media = [9428.12, 8970.49, 8970.49]
#
# # Crearea plotului
# plt.plot(nr_iteratii, best, label='Best')
# plt.plot(nr_iteratii, media, label='Media')
#
# # Adăugarea etichetelor și titlului
# plt.xlabel('Nr. Iteratii')
# plt.ylabel('Valoare')
# plt.title('Grafic Best vs. Media')
# plt.legend()
#
# # Afișarea plotului
# plt.show()

import matplotlib.pyplot as plt

# Datele pentru coloana 3 ("best") și coloana 4 ("media")
best = [81108.41, 34851.80, 26548.42, 24291.49]
media = [84995.95, 36268.60, 27785.51, 25114.01]

# Numărul de iterații
# iteratii = list(range(1, 11))
iteratii = [10, 20, 50, 100]
# Crearea graficului
plt.plot(iteratii, best, label='Best')
plt.plot(iteratii, media, label='Media')

# Adăugarea titlului și etichetelor pentru axele x și y
plt.title('Compararea Best și Media')
plt.xlabel('Iterații')
plt.ylabel('Valori')

# Adăugarea unei legende
plt.legend()

# Afișarea graficului
plt.show()
