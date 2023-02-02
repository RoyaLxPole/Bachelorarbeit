import copy
import help
import numpy as np
from tqdm import tqdm

import large_random_graph_generator


def run_graph_generator_with_file(number_of_nodes):
    matrix_collection, possible_coloring = read_matrix_from_file('n={}.txt'.format(number_of_nodes), number_of_nodes, colored_by_backtracking=True)
    for i in tqdm(range(len(matrix_collection))):
        graph_generator(matrix_collection[i], number_of_nodes + 1)

def graph_generator(matrix, n):
    filename = 'n={}_test_forcliquetest.txt'.format(n)
    matrix = extend_matrix(matrix)
    binary_matrix = create_binary_matrix((pow(2, len(matrix) - 1), (len(matrix) - 1)))

    for i in range(len(binary_matrix) - 1):
        matrix_copy = copy.deepcopy(matrix)
        create_new_graph(matrix_copy, binary_matrix[i], filename)
    #begrenzung auf 100 größtmögliche Graphen
    '''for i in range(100):
        matrix_copy = copy.deepcopy(matrix)
        create_new_graph(matrix_copy, binary_matrix[i], filename)'''

def create_new_graph(matrix, binary_matrix_list, filename):
    for i in range(len(binary_matrix_list)):
        if binary_matrix_list[i] == 1:
            matrix = add_edge(matrix, i)

    write_matrix_in_file(matrix, filename)


def add_edge(matrix_copy, index):
    for i in range(0, len(matrix_copy)):
        if matrix_copy[i][index] > 0:
            matrix_copy[i][index] += 1

    matrix_copy[len(matrix_copy) - 1][index] = matrix_copy[index][index]
    matrix_copy[len(matrix_copy) - 1][len(matrix_copy) - 1] += 1

    for i in range(len(matrix_copy)):
        if matrix_copy[len(matrix_copy) - 1][i] > 0:
            matrix_copy[i][len(matrix_copy) - 1] = matrix_copy[len(matrix_copy) - 1][len(matrix_copy) - 1]

    return matrix_copy


def extend_matrix(matrix):
    column_to_be_added = np.zeros(len(matrix))
    row_to_be_added = np.zeros(len(matrix) + 1)
    result = np.column_stack((matrix, column_to_be_added))
    result = np.row_stack((result, row_to_be_added))
    return result


def create_binary_matrix(shape):
    binary_matrix = np.zeros(shape)
    increment_list = [False] * shape[1]
    range_list = []
    counter_list = [np.power(2, i) for i in reversed(range(shape[1]))]

    for i in reversed(range(shape[1])):
        to_append = np.power(2, i)
        range_list.append(to_append)

    for i in range(shape[0]):
        for j in range(shape[1]):
            if increment_list[j]:
                counter_list[j] = counter_list[j] + 1
            if not increment_list[j]:
                counter_list[j] = counter_list[j] - 1
            if not increment_list[j]:
                binary_matrix[i][j] = 1
            if counter_list[j] == 0:
                increment_list[j] = True
            if counter_list[j] == range_list[j]:
                increment_list[j] = False

    return binary_matrix


def write_matrix_in_file(matrix, filename):
    possible_coloring = []
    yes_no = False
    yes_no_4_clique = find_4_clique_in_graph(matrix)
    if not yes_no_4_clique:
        yes_no, possible_coloring = backtracking(matrix)
    if yes_no:
        file = open(filename, 'a')

        for i in range(len(matrix)):
            matrix_line = np.array2string(matrix[i].astype(np.int), max_line_width=500)
            file.write(matrix_line)
            file.write('\n')

        file.write(' '.join(str(e) for e in possible_coloring))
        file.write('\n')
        file.write('\n')
        file.close()


def read_matrix_from_file(filename, matrix_size, colored_by_backtracking):
    file = open(filename, 'r')
    line_count = 0
    for line in file:
        if line != "\n":
            line_count += 1
    file.close()
    print(line_count)
    file = open(filename, 'r')
    if colored_by_backtracking:
        matrix_collection = np.zeros((line_count // (matrix_size + 1),matrix_size,matrix_size), dtype=np.int8)
    else:
        matrix_collection = np.zeros((line_count // matrix_size, matrix_size, matrix_size), dtype=np.int8)
    colors = ['r', 'b', 'g']
    possible_coloring = []
    counter_for_matrix_out = 0
    counter_for_matrix_in = 0
    for line in file:
        flag = False
        if colored_by_backtracking:
            for color in colors:
                if color in line:
                    line_list = list(line.split())
                    possible_coloring.append(list(line_list))
                    flag = True
                    break
        if flag:
            continue
        if line != '\n':
            line_list = list(line.split())
            line_list[0] = line_list[0][1:]
            line_list[len(line_list) - 1] = line_list[len(line_list) - 1][:-1]
            matrix_collection[counter_for_matrix_out][counter_for_matrix_in] = np.asarray(list(map(int, line_list)))
            '''print('new Matrix')
            print(matrix_collection[counter_for_matrix_out])
            print(counter_for_matrix_out)
            print(counter_for_matrix_in)'''
            counter_for_matrix_in += 1
        if line == '\n':
            counter_for_matrix_in = 0
            counter_for_matrix_out += 1
    file.close()
    if colored_by_backtracking:
        return matrix_collection, possible_coloring
    return matrix_collection

def read_matrix_and_run_my_algorithm(filename, size, colored_by_backtracking):
    file = open('with_counter_solved\myAlg_optim_solve_{}'.format(size), 'a')
    not_solve_file = open('with_counter_not_solved\myAlg_optim_not_solve_{}'.format(size), 'a')
    if colored_by_backtracking:
        matrix_collection, coloring_collection = read_matrix_from_file(filename, size, colored_by_backtracking)
    else:
        matrix_collection = read_matrix_from_file(filename, size, colored_by_backtracking)
    sum_colloring_time = 0
    counter = 0
    if colored_by_backtracking:
        for m in tqdm(range(len(matrix_collection))):
            matrix = matrix_collection[m]
            coloring = coloring_collection[m]
            counter += 1
            matrix_copy = copy.deepcopy(matrix)
            #print(counter)
            colors, matrix_sorted, coloring_time, loop_counter = help.run_sort_matrix(matrix_copy)
            sum_colloring_time += coloring_time
            if 0 in colors:
                for i in range(len(matrix)):
                    matrix_line = np.array2string(matrix[i].astype(np.int), max_line_width=500)
                    not_solve_file.write(matrix_line)
                    not_solve_file.write('\n')
                not_solve_file.write(' '.join(str(e) for e in coloring))
                not_solve_file.write('\n')
                not_solve_file.write('\n')

                for i in range(len(matrix_sorted)):
                    matrix_line = np.array2string(matrix[i].astype(np.int), max_line_width=500)
                    not_solve_file.write(matrix_line)
                    not_solve_file.write('\n')

                not_solve_file.write(' '.join(str(e) for e in colors))
                not_solve_file.write('\n')
                not_solve_file.write('\n')
                continue

            for i in range(len(matrix_sorted)):
                matrix_line = np.array2string(matrix[i].astype(np.int), max_line_width=500)
                file.write(matrix_line)
                file.write('\n')

            file.write(' '.join(str(e) for e in colors))
            file.write('\n')
            file.write('counter_loop: {}'.format(loop_counter))
            file.write('\n')
            file.write('\n')


            #print(matrix_sorted)
            #print(colors)
        file.close()
        not_solve_file.close()
        return sum_colloring_time
    else:
        for m in tqdm(range(len(matrix_collection))):
            print('itsa me mario')
            matrix = matrix_collection[m]
            counter += 1
            matrix_copy = copy.deepcopy(matrix)
            # print(counter)
            colors, matrix_sorted, coloring_time = help.run_sort_matrix(matrix_copy)
            print(colors)
            sum_colloring_time += coloring_time
            if 0 in colors:
                print('here')
                for i in range(len(matrix_sorted)):
                    matrix_line = np.array2string(matrix[i].astype(np.int), max_line_width=500)
                    not_solve_file.write(matrix_line)
                    not_solve_file.write('\n')

                not_solve_file.write(' '.join(str(e) for e in colors))
                not_solve_file.write('\n')
                not_solve_file.write('\n')
                continue

            for i in range(len(matrix_sorted)):

                matrix_line = np.array2string(matrix[i].astype(np.int), max_line_width=500)
                file.write(matrix_line)
                file.write('\n')

            file.write(' '.join(str(e) for e in colors))
            file.write('\n')
            file.write('\n')
            print('motherfucker')
            # print(matrix_sorted)
            # print(colors)
        print('here we go again')
        file.close()
        not_solve_file.close()
        return sum_colloring_time


def sort_matrix_files(filename, is_in_normal_form):
    file = open(filename, 'r')
    colors = ['r', 'b', 'g']
    possible_coloring = []
    counter_for_matrix_in = 0
    flag_matrix_not_symmetric = False
    matrix = np.zeros(0, dtype=np.int8)
    file_open = False
    for line in file:
        flag = False
        for color in colors:
            if color in line:
                line_list = list(line.split())
                possible_coloring.append(line_list)
                flag = True
                break
        if flag:
            continue
        if line != '\n':

            if not file_open:
                splitter = line.split()
                matrix_length = len(splitter)
                write_file = open('Graphen\\internet_Graphs\\internet_Graphs-sorted\\{}_sorted_internet_Graphs.txt'.format(matrix_length), 'a')
                file_open = True
                matrix = np.zeros((matrix_length,matrix_length))
            line_list = list(line.split())
            #line_list[0] = line_list[0][1:]
            #line_list[len(line_list) - 1] = line_list[len(line_list) - 1][:-1]
            ar = np.asarray(list(map(int, line_list)), dtype=np.int8)
            matrix[counter_for_matrix_in] = ar
            counter_for_matrix_in += 1
        if line == '\n':
            matrix = matrix.astype('int')
            if is_in_normal_form:
                matrix = large_random_graph_generator.generate_adj_matrix_my_form(matrix)
            for i in range(len(matrix)):

                matrix_line = np.array2string(matrix[i].astype(np.int), max_line_width=500)
                write_file.write(matrix_line)
                write_file.write('\n')

            #write_file.write(' '.join(str(e) for e in possible_coloring[0]))
            write_file.write('\n')
            write_file.write('\n')
            write_file.close()

            counter_for_matrix_in = 0
            possible_coloring = []
            file_open = False


def backtracking(matrix):
    matrix = matrix.astype(int)

    g = help.Graph(len(matrix))
    g.graph = matrix


    yes_no, possible_coloring = g.graphColouring(3)
    return yes_no, possible_coloring


def read_file_and_do_backtracking(filename, matrix_size):
    matrix_collection, _ = read_matrix_from_file(filename, matrix_size)
    for matrix in matrix_collection:
        backtracking(matrix)


def find_4_clique_in_graph(matrix):
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            if matrix[i][j] >= 3:
                for k in range(j + 1, len(matrix)):
                    if matrix[i][k] >= 3:
                        if matrix[j][k] >=3:
                            for l in range(k + 1, len(matrix)):
                                if matrix[i][l] >= 3:
                                    if matrix[j][l] >= 3:
                                        if matrix[k][l] >= 3:
                                            return True
    return False


def get_matrix_size_from_file(filename):
    file = open(filename, 'r')
    line_count = 0
    for line in file:
        if line != "\n":
            line_count += 1
        if line == '\n':
            file.close()
            return line_count - 1
