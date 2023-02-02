import copy

import numpy as np
import graph_generator as gg
import help
import help as h
import os
import scandir
from tqdm import tqdm



def main():
    number_of_nodes = [10]
    for i in number_of_nodes:
        gg.run_graph_generator_with_file(i)

def main2():
    files = ['11-21.txt','22-29.txt', '30-37.txt', '38-72.txt', '73.txt']
    for file in files:
        gg.sort_matrix_files('Graphen\\internet_Graphs\\{}'.format(file), True)

def main3():
    size = 14
    gg.read_matrix_and_run_my_algorithm('Graphen\\internet_Graphs\\internet_Graphs-sorted\\{}_sorted_internet_Graphs.txt'.format(size), size, colored_by_backtracking=False)


def test_bucket_sort():
    matrix = np.array([[2, 4, 5, 0, 0, 0, 0],
                       [2, 4, 5, 3, 0, 0, 2],
                       [2, 4, 5, 3, 1, 1, 0],
                       [0, 4, 5, 3, 0, 0, 2],
                       [0, 0, 5, 0, 1, 0, 0],
                       [0, 0, 5, 0, 0, 1, 0],
                       [0, 4, 0, 3, 0, 0, 2]])
    original_matrix = copy.deepcopy(matrix)
    list = np.arange(0, len(original_matrix), 1, dtype=int)
    help.bucket_sort(matrix, original_matrix, list, 0)

def main_run_sort_matrix():
    matrix = np.array([[2, 4, 5, 0, 0, 0, 0],
                       [2, 4, 5, 3, 0, 0, 2],
                       [2, 4, 5, 3, 1, 1, 0],
                       [0, 4, 5, 3, 0, 0, 2],
                       [0, 0, 5, 0, 2, 2, 0],
                       [0, 0, 5, 0, 2, 2, 0],
                       [0, 4, 0, 3, 0, 0, 2]])
    couldnsovlematrix = np.array([[6, 5, 4, 3, 3, 3, 3, 0],
                                  [6, 5, 4, 3, 3, 0, 0, 3],
                                  [6, 5, 4, 0, 0, 3, 0, 3],
                                  [6, 5, 0, 3, 0, 3, 0, 0],
                                  [6, 5, 0, 0, 3, 0, 3, 0],
                                  [6, 0, 4, 3, 0, 3, 0, 0],
                                  [6, 0, 0, 0, 3, 0, 3, 3],
                                  [0, 5, 4, 0, 0, 0, 3, 3]])

    solvethis = np.array([[3,3,3,0,0,0,0,0,0,0,3,0],
                          [3,3,3,3,0,0,0,0,0,0,0,0],
                          [3,3,3,3,0,0,0,0,0,0,0,0],
                          [0,3,3,3,3,0,0,0,0,0,0,0],
                          [0,0,0,3,3,3,3,0,0,0,0,0],
                          [0,0,0,0,3,3,3,3,0,0,0,0],
                          [0,0,0,0,3,3,3,3,0,0,0,0],
                          [0,0,0,0,0,3,3,3,3,0,0,0],
                          [0,0,0,0,0,0,0,3,3,3,0,3],
                          [0,0,0,0,0,0,0,0,3,3,3,3],
                          [3,0,0,0,0,0,0,0,0,3,3,3],
                          [0,0,0,0,0,0,0,0,3,3,3,3,]])
    colors, original_matrix, coloring_time, counter = help.run_sort_matrix(solvethis)
    print(colors)
    print(original_matrix)
    print(counter)

def main_run_in_data():
    gg.read_matrix_and_run_my_algorithm('Graphen_optim/n=11.txt', 11, True)


if __name__ == "__main__":
    #test_bucket_sort()
    #main_run_sort_matrix()
    main_run_in_data()
    '''folder = 'Graphen_optim'  # here your dir path
    print("All files ending with .py in folder %s:" % folder)

    file_list = []
    matrix_file_size = []
    for paths, dirs, files in scandir.walk(folder):
        # for (paths, dirs, files) in os.walk(folder):
        for file in files:
            if file.endswith(".txt"):
                file_list.append(os.path.join(paths, file))
    for i in range(len(file_list)):
        matrix_file_size.append(gg.get_matrix_size_from_file(file_list[i]))
    print(len(file_list), file_list)
    print(matrix_file_size)
    for i in tqdm(range(len(file_list))):
        gg.read_matrix_and_run_my_algorithm(file_list[i], matrix_file_size[i], True)'''
