import numpy as np
from basic_operations import choose_mode
from typing import List, AnyStr

X = List[List[float]]


def get_x_with_index(neighbors):
    return list(enumerate(neighbors))


def get_closest_neighbors(x: List, k: int):
    neighbors = get_x_with_index(x)
    neighbors.sort(key=lambda tup: tup[1])
    return neighbors[1:k + 1]


def calculate_density(closest_neighbors, k: int):
    values = list(map(lambda tup: tup[1], closest_neighbors))
    return 1 / ((1 / k) * sum(values))


def calculate_ard(i: int, x_string: AnyStr, k: int):
    x = np.array(choose_mode(x_string))
    closest_neighbors = get_closest_neighbors(x[i], k)
    density = calculate_density(closest_neighbors, k)
    ard_sum = 0
    for neighbor in closest_neighbors:
        (index, _) = neighbor
        closest_neighbors = get_closest_neighbors(x[index], k)
        ard_sum = ard_sum + calculate_density(closest_neighbors, k)

    return density / ((1 / k) * ard_sum)


# string_table = """O1 O2 O3 O4 O5 O6 O7 O8 O9 O10
# O1 0 8.55 0.43 1.25 1.14 3.73 2.72 1.63 1.68 1.28
# O2 8.55 0 8.23 8.13 8.49 6.84 8.23 8.28 8.13 7.66
# O3 0.43 8.23 0 1.09 1.10 3.55 2.68 1.50 1.52 1.05
# O4 1.25 8.13 1.09 0 1.23 3.21 2.17 1.29 1.33 0.56
# O5 1.14 8.49 1.10 1.23 0 3.20 2.68 1.56 1.50 1.28
# O6 3.73 6.84 3.55 3.21 3.20 0 2.98 2.66 2.50 3.00
# O7 2.72 8.23 2.68 2.17 2.68 2.98 0 2.28 2.30 2.31
# O8 1.63 8.28 1.50 1.29 1.56 2.66 2.28 0 0.25 1.46
# O9 1.68 8.13 1.52 1.33 1.50 2.50 2.30 0.25 0 1.44
# O10 1.28 7.66 1.05 0.56 1.28 3.00 2.31 1.46 1.44 0"""
#
# print(calculate_ard(i=1, x_string=string_table, k=2))

string_table2 = '5.7 6.0 6.2 6.3 6.4 6.6 6.7 6.9 7.0 7.4'

print(calculate_ard(i=9, x_string=string_table2, k=3))
