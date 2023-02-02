import networkx as nx
import numpy as np
import help
import graph_generator as gg
from tqdm import tqdm

def generate_random_graph(n, probability, looping):
    for i in tqdm(range(looping)):
        graph = nx.fast_gnp_random_graph(n, probability)
        A = nx.to_numpy_array(graph)
        A = A.astype(int)
        matrix = generate_adj_matrix_my_form(A)
        visited = set()
        if len(matrix) > 0 and help.check_connected_it(matrix):
            gg.write_matrix_in_file(matrix, 'n={}tryal.txt'.format(n))


def generate_adj_matrix_my_form(matrix):
    diagonale = np.sum(matrix, axis=1)

    for i in range(len(matrix)):
        matrix[i][i] = diagonale[i]
        for j in range(len(matrix)):
            if matrix[i][j] > 0:
                matrix[i][j] = diagonale[j]
    i = 0
    while True:

        if matrix[i][i] == 0:
            matrix = help.delete_row(matrix, i)
            i = i-1
        i = i + 1
        if i >= len(matrix):
            break
    return matrix