import copy
import numpy as np
import graph_generator as gg
import time
def run_sort_matrix(matrix):
    original_matrix = copy.deepcopy(matrix)
    list = np.arange(0, len(original_matrix), 1, dtype=int)

    for i in range(len(original_matrix)):
        counter = 1
        visited = set()
        connected = False
        #print('before sort matrix')
        #matrix, original_matrix, list = sort_matrix(matrix, original_matrix, list, i)
        matrix, original_matrix, list = bucket_sort(matrix, original_matrix, list, i)
        #print('after sort matrix')
        matrix_backup = copy.deepcopy(matrix)
        matrix = delete_first_row(matrix)
        #print('check connection')
        if check_connected(matrix):
            connected = True
        while not connected and counter < len(matrix):
            matrix = copy.deepcopy(matrix_backup)
            matrix, original_matrix, list = delete_row_in_matrix_with_index(matrix, original_matrix, list, counter, i)
            counter += 1
            connected = check_connected(matrix)
        #print('orginal')
        #print(original_matrix)
        #print('schrumpfend')
        #print(matrix)
        #print(list)
    st = time.process_time()
    #colors = color_graph_my_alg(original_matrix)
    #print('run_sort_matrix before coloring')
    colors, counter = color_graph_my_alg_optim1(original_matrix)
    et = time.process_time()
    coloring_time = et - st
    #print('coloringtime {}'.format(coloring_time))
    #print(colors)
    return colors, original_matrix, coloring_time, counter

def delete_row_in_matrix_with_index(matrix, original_matrix, list, index, deleted):
    swap(deleted, deleted + index, original_matrix)
    swap_list(list, deleted, deleted + index)
    matrix = delete_row(matrix, index)
    return matrix, original_matrix, list


def delete_first_row(matrix):
    for i in range(1, len(matrix)):
        matrix[:, 0] = 0
        if matrix[0][i] != 0:
            for j in range(len(matrix)):
                if matrix[j][i] > 0:
                    matrix[j][i] = matrix[j][i] - 1
    copy_matrix = copy.deepcopy(matrix[1:, 1:])
    return copy_matrix

def delete_row(matrix, index):
    matrix = swap(0, index, matrix)
    matrix = delete_first_row(matrix)
    return matrix

def swap_list(list, i, j):
    temp = list[i]
    list[i] = list[j]
    list[j] = temp


def sort_matrix(matrix, original_matrix, list, deleted):
    for i in range(len(matrix)):
        k = i
        j = i - 1
        while j >= 0 and matrix[k][k] < matrix[j][j]:
            swap(k, j, matrix)
            swap(k + deleted, j+deleted, original_matrix)
            swap_list(list, k + deleted, j + deleted)
            j = j - 1
            k = k - 1
        while j >= 0 and matrix[k][k] == matrix[j][j]:
            if is_in_deep_less_relevant(k, j, matrix):
                swap(k, j, matrix)
                swap(k + deleted, j + deleted, original_matrix)
                swap_list(list, k + deleted, j + deleted)
            j = j - 1
            k = k - 1
    return matrix, original_matrix, list


def color_graph_my_alg(matrix):
    k = 0
    while k < len(matrix):
        flag = False
        coloring = np.zeros(len(matrix))
        coloring = coloring.tolist()
        coloring[len(matrix) - k - 1] = 'r'

        for i in reversed(range(len(matrix) - k - 1)):
            matching = ['r', 'b', 'g']
            for j in range(i, len(matrix)):
                if matrix[i][j] > 0 and coloring[j] in matching:
                    matching.remove(coloring[j])
            if matching:
                coloring[i] = matching[0]
            else:
                flag = True
                break
        if flag:
            k += 1
            continue
        #verbleibende Knoten färben
        for i in range(len(matrix)-k, len(matrix)):
            matching = ['r', 'b', 'g']
            for j in range(0, len(matrix)):
                if matrix[i][j] > 0 and coloring[j] in matching:
                    matching.remove(coloring[j])
            if matching:
                coloring[i] = matching[0]
            else:
                flag = True
                break
        if flag:
            k += 1
            continue
        if 0 not in coloring:
            return coloring
    return np.zeros(len(matrix))

def color_graph_my_alg_optim1(matrix):
    counter_list = np.full(len(matrix), 3)
    coloring_list = np.full((len(matrix),3),['r', 'b', 'g'])
    coloring_list = coloring_list.tolist()
    #print(counter_list)
    #print(coloring_list)
    k = 0
    while k < len(matrix):
        flag = False
        counter = copy.deepcopy(counter_list)
        coloring = copy.deepcopy(coloring_list)
        coloring, counter = reduce_counter_and_delete_color_in_neighbours_list(len(matrix) - k - 1, matrix, 'r', coloring, counter, True)
        already_colored = []
        already_colored.append(len(matrix) - k - 1)
        for i in reversed(range(len(matrix) - k - 1)):
            #TODO man kann hier nicht mehr so einfach durch iterieren deswegen findet er in matching keine Farbe mehr die er zu weisen kann weswegen der Code kein ergebnis liefert
            #print('This is i: {}'.format(i))
            #print('This is k: {}'.format(k))
            if i in already_colored:
                continue

            if coloring[i]:
                coloring, counter = reduce_counter_and_delete_color_in_neighbours_list(i, matrix, coloring[i][0], coloring, counter)

                counter_indices_with_one = np.argwhere(counter == 1)
                counter_indices_with_one = np.squeeze(counter_indices_with_one)
                counter_list_indices_with_one = counter_indices_with_one.tolist()

                while counter_list_indices_with_one:

                    cliwo_right_order = counter_indices_with_one[counter_indices_with_one <= len(matrix) - k - 1]
                    cliwo_right_order_list = cliwo_right_order.tolist()
                    cliwo_left_element = counter_indices_with_one[counter_indices_with_one > len(matrix) - k - 1]
                    cliwo_left_element_list = cliwo_left_element.tolist()
                    if cliwo_right_order_list:
                        coloring, counter = reduce_counter_and_delete_color_in_neighbours_list(cliwo_right_order[-1], matrix, coloring[cliwo_right_order[-1]][0], coloring, counter)
                        already_colored.append(cliwo_right_order[-1])
                        counter_list_indices_with_one = np.argwhere(counter == 1)
                        counter_indices_with_one = np.squeeze(counter_list_indices_with_one)
                        counter_list_indices_with_one = counter_indices_with_one.tolist()
                        if 0 in counter:
                            flag = True
                        continue
                    if cliwo_left_element_list:
                        coloring, counter = reduce_counter_and_delete_color_in_neighbours_list(cliwo_left_element[-1],matrix, coloring[cliwo_left_element[-1]][0],coloring, counter)
                        already_colored.append(cliwo_left_element[-1])
                        counter_list_indices_with_one = np.argwhere(counter == 1)
                        counter_indices_with_one = np.squeeze(counter_list_indices_with_one)
                        counter_list_indices_with_one = counter_indices_with_one.tolist()
                        if 0 in counter:
                            flag = True
                        continue

            else:
                flag = True
                break


        for i in range(len(matrix)-k, len(matrix)):

            if i in already_colored:
                continue

            if coloring[i]:
                coloring, counter = reduce_counter_and_delete_color_in_neighbours_list(i, matrix, coloring[i][0], coloring, counter)
                counter_list_indices_with_one = np.argwhere(counter == 1)
                counter_indices_with_one = np.squeeze(counter_indices_with_one)
                counter_list_indices_with_one = counter_indices_with_one.tolist()
                while counter_list_indices_with_one:

                    cliwo_right_order = counter_list_indices_with_one[counter_list_indices_with_one <= len(matrix) - k - 1]
                    cliwo_left_element = counter_list_indices_with_one[counter_list_indices_with_one > len(matrix) - k - 1]
                    counter_indices_with_one = np.squeeze(counter_indices_with_one)
                    counter_list_indices_with_one = counter_indices_with_one.tolist()
                    if cliwo_right_order:
                        coloring, counter = reduce_counter_and_delete_color_in_neighbours_list(cliwo_right_order[-1], matrix, coloring[cliwo_right_order[-1]][0], coloring, counter)
                        already_colored.append(cliwo_right_order[-1])
                        counter_list_indices_with_one = np.argwhere(counter == 1)
                        counter_indices_with_one = np.squeeze(counter_list_indices_with_one)
                        counter_list_indices_with_one = counter_indices_with_one.tolist()
                        if 0 in counter:
                            flag = True
                        continue
                    if cliwo_left_element:
                        coloring, counter = reduce_counter_and_delete_color_in_neighbours_list(cliwo_left_element[-1],matrix, coloring[cliwo_left_element[-1]][0],coloring, counter)
                        already_colored.append(cliwo_left_element[-1])
                        counter_list_indices_with_one = np.argwhere(counter == 1)
                        counter_indices_with_one = np.squeeze(counter_list_indices_with_one)
                        counter_list_indices_with_one = counter_indices_with_one.tolist()
                        if 0 in counter:
                            flag = True
                        continue
            else:
                flag = True
                break
        if flag:
            k += 1
            continue


        if np.all(counter == np.full((len(counter)),-1)):
            return coloring, k
    return np.zeros(len(matrix), dtype=np.int), k


def reduce_counter_and_delete_color_in_neighbours_list(index, matrix, color, coloring, counter, initial=False):
    if initial:
        coloring[index] = ['r']
        counter[index] = -1
        for i in range(len(matrix)):
            if i == index or matrix[index][i] == 0:
                continue
            if color in coloring[i]:
                coloring[i].remove(color)
                counter[i] -= 1
        return coloring, counter
    coloring[index] = [color]
    counter[index] = -1
    for i in range(len(matrix)):
        if i == index or matrix[index][i] == 0:
            continue
        if color in coloring[i] and len(coloring[i]) >= 1:
            coloring[i].remove(color)
            counter[i] -= 1
    return coloring, counter



def is_in_deep_less_relevant(j, i, matrix):
    array1 = find_min(j, matrix, [])
    array2 = find_min(i, matrix, [])
    k = 0
    sum1 = 0
    sum2 = 0

    while sum1 == sum2 and k < min([len(array1) ,len(array2)]) and array1[k] != array2[k]:
        if array1[k] == array2[k]:
            break
        if array1[k] == np.inf:
            return False
        if array2[k] == np.inf:
            return True
        sum1 = sum1 + matrix[array1[k]][array1[k]]
        sum2 = sum2 + matrix[array2[k]][array2[k]]
        k = k + 1
    lastindex1 = np.inf
    lastindex2 = np.inf
    if len(array1) < len(array2):
        array2 = array2[:len(array1)]
    if len(array2) < len(array1):
        array1 = array1[:len(array2)]
    while sum1 == sum2 and len(array1) < len(matrix) and array1[-1] != array2[-1]:
        array1 = find_min(array1[-1], matrix, array1)
        array2 = find_min(array2[-1], matrix, array2)
        if lastindex1 == array2[-1] or lastindex2 == array1[-1]:
            break
        lastindex1 = array1[-1]
        lastindex2 = array2[-1]
        if lastindex1 == lastindex2:
            break
        if lastindex1 == np.inf:
            return False
        if lastindex2 == np.inf:
            return True
        sum1 += matrix[array1[-1]][array1[-1]]
        sum2 += matrix[array2[-1]][array2[-1]]
    if sum1 < sum2:
        return True
    return False

def find_min(index, matrix, visited):
    if index == np.inf:
        return visited
    if not(index in visited):
        visited_new = visited.copy()
        visited_new.append(index)
    if index in visited:
        visited_new = visited.copy()
    counter = 0
    min = np.inf
    for i in range(len(matrix)):
        #print('i, index, matrixf')
        #print(i)
        #print(index)
        #print(matrix[index][i])
        if i == index or matrix[index][i] == 0 or i in visited:
            continue
        if matrix[index][i] == min:
            counter = counter + 1
            index_array.append(i)
        if matrix[index][i] < min:
            min = matrix[index][i]
            counter = 1
            index_array = [i]
    if counter == 0:
        visited_new.append(np.inf)
        return visited_new
    #brauche ein Ende für den Fall das es keine weiteren Knoten gibt
    if counter == 1:
        visited_new.append(index_array[0])
        return visited_new
    if counter > 1:
        final_visited_list = find_min(index_array[0], matrix, visited_new)
        for i in range(1, len(index_array)):
            help_visited = find_min(index_array[i], matrix, visited_new)
            j = 0
            fvl_length = len(final_visited_list)
            hv_length = len(help_visited)
            while j < np.min([fvl_length, hv_length]):
                if final_visited_list[j] == np.inf or help_visited[j] == np.inf:
                    j = j + 1
                    continue
                if matrix[final_visited_list[j]][final_visited_list[j]] < matrix[help_visited[j]][help_visited[j]]:
                    break
                if matrix[final_visited_list[j]][final_visited_list[j]] > matrix[help_visited[j]][help_visited[j]]:
                    final_visited_list = help_visited
                    break
                j = j + 1
        return final_visited_list
    return visited

def get_sum(index_array, matrix):
    sum = 0
    for i in index_array:
        sum += matrix[i][i]
    return sum

def find_min_path(index, matrix):
    visited = np.array(index)
    sums = np.array(matrix[index][index])
    minimum, min_index = find_min(index, matrix, visited)
    if len(min_index) == 1:
        sums.append(minimum)
        visited.append(min_index)
        return visited, sums
    if len(min_index) > 1:
        pass





def find_min_one(index, matrix, visited):
    minimum = np.inf
    min_index = np.array()
    for i in range(len(matrix)):
        if i != index and matrix[index][i] > 0 and i not in visited:
            if matrix[index][i] < minimum:
                minimum = matrix[index][i]
                min_index = np.array(i)
            if matrix[index][i] == minimum:
                min_index.append(i)
    return minimum, min_index





def swap(index1, index2, matrix):
    matrix[[index2, index1], :] = matrix[[index1, index2], :]
    matrix[:, [index2, index1]] = matrix[:, [index1, index2]]
    return matrix

def check_connected(matrix):
    queue = []
    queue.append(0)
    visited = set()
    while queue:
        q = queue.pop(0)
        visited.add(q)
        for i in range(len(matrix)):
            if i == q:
                continue
            if matrix[q][i] > 0 and i not in visited and i not in queue:
                queue.append(i)
    if len(visited) != len(matrix):
        return False
    return True


def check_connected_it(matrix):
    visited = set()
    queue = set()
    queue.add(0)
    while len(queue):
        current = queue.pop()
        visited.add(current)
        for j in range(len(matrix)):
            if matrix[current][j]:
                if j in queue or j in visited:
                    continue
                else:
                    queue.add(j)

    if len(visited) != len(matrix):
        #print(False)
        return False
    #print(True)
    return True

def check_node_neighbors_coloring(color1, color2):
    #print(color1)
    #print(color2)
    if color1 != color2:
        return True
    else:
        return False


def check_coloring_graph(coloring, matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if i != j and matrix[i][j] > 0:
                flag = check_node_neighbors_coloring(coloring[i], coloring[j])
                if not flag:
                    print('there is a error in the coloring')
    print('3-coloring is right')



class Graph():

    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)] \
                      for row in range(vertices)]

    # A utility function to check
    # if the current color assignment
    # is safe for vertex v
    def isSafe(self, v, colour, c):
        for i in range(self.V):
            if self.graph[v][i] > 0 and colour[i] == c:
                return False
        return True

    # A recursive utility function to solve m
    # coloring  problem
    def graphColourUtil(self, m, colour, v):
        if v == self.V:
            return True

        for c in range(1, m + 1):
            if self.isSafe(v, colour, c) == True:
                colour[v] = c
                if self.graphColourUtil(m, colour, v + 1) == True:
                    return True
                colour[v] = 0

    def graphColouring(self, m):
        colour = [0] * self.V
        if self.graphColourUtil(m, colour, 0) == None:
            return False, None

        coloring = ['r', 'b', 'g']
        pos_coloring = []
        for i in colour:
            pos_coloring.append(coloring[i - 1])
        return True, pos_coloring

def check_file_for_coloring(filename, matrix_size):
    matrix_collection,coloring_collection = gg.read_matrix_from_file(filename, matrix_size)
    for m in range(len(matrix_collection)):
        coloring = coloring_collection[m]
        matrix = matrix_collection[m]
        flag = False
        for i in range(len(matrix)):
            color = coloring[i]
            for j in range(len(matrix)):
                if j != i and matrix[i][j] > 0:
                    if color == coloring[j]:
                        print('This Graph is not 3 Colorable')
                        #print(matrix)
                        #print('\n')
            if flag:
                break
        if flag:
            continue


def bucket_sort(matrix, original_matrix, list, deleted):
    highest_grade = find_highest_grade(matrix)
    buckets = [[] for _ in range(highest_grade)]
    if len(matrix) <= 1:
        #print('why')
        #print('not')
        return matrix, original_matrix, list
    for i in range(len(matrix)):
        buckets[matrix[i][i] - 1].append((matrix[i], i))
    for i in range(len(buckets)):
        if len(buckets[i]) > 1:
            buckets[i] = sort_bucket_inside(matrix, buckets[i])
    counter = 0
    swaps = [[] for _ in range(len(matrix))]
    for i in range(len(buckets)):
        for j in range(len(buckets[i])):
            swaps[counter] = buckets[i][j][1]
            counter += 1
    #print(swaps)
    return_matrix = np.zeros(matrix.shape)
    for i in range(len(swaps)):

        swap(i, swaps[i], matrix)
        swap(i + deleted, swaps[i] + deleted, original_matrix)
        swap_list(list, i + deleted, swaps[i] + deleted)
        index = swaps.index(i)
        swaps[index] = swaps[i]
    return matrix, original_matrix, list



#trash
def sort_bucket_inside(matrix, bucket):

    '''smallest_paths = [[] for _ in range(len(bucket))]
    for i in range(len(bucket)):
        smallest_paths[i] = find_complete_minimal_path(matrix, bucket[i][1])
    print(smallest_paths)
    bucket = sort_bucket_in_order_of_relevance(matrix, bucket, smallest_paths, highest_grade)'''

    for i in reversed(range(len(bucket))):
        k = i
        j = i - 1
        while j >= 0:
            if is_in_deep_less_relevant(bucket[k][1], bucket[j][1], matrix):
                swap_element = bucket[k]
                bucket[k] = bucket[j]
                bucket[j] = swap_element
            k -= 1
            j -= 1
    return bucket



def find_highest_grade(matrix):
    highest_grade = 0
    for i in range(len(matrix)):
        if highest_grade < matrix[i][i]:
            highest_grade = matrix[i][i]
    return highest_grade


def find_complete_minimal_path(matrix, node_index):
    visited = [node_index]
    while visited[-1] != np.inf:
        visited = find_min(visited[-1], matrix, visited)
    return visited

def sort_bucket_in_order_of_relevance(matrix, bucket, smallest_paths, highest_grade):
    smallest_paths_length = [len(x) for x in smallest_paths]
    smallest_paths_grades = [[] for _ in range(len(smallest_paths))]
    for i in range(len(smallest_paths_grades)):
        smallest_paths_grades[i] = [[] for _ in range(len(smallest_paths[i]))]
    for i in range(len(smallest_paths)):
        for j in range(len(smallest_paths[i])):
            if smallest_paths[i][j] == np.inf:
                smallest_paths_grades[i][j] = np.inf
                continue
            smallest_paths_grades[i][j] = matrix[smallest_paths[i][j]][smallest_paths[i][j]]
    #print(smallest_paths_length)
    for i in range(np.max(smallest_paths_length)):
        bucket_length = [[] for _ in range(highest_grade + 1)]
        dynamic_bucket = [[] for _ in range(highest_grade + 1)]
        already_in_right_bucket = []
        for j in range(len(bucket)):
            if j in already_in_right_bucket:
                continue
            if smallest_paths_grades[j][i] == np.inf:
                dynamic_bucket[-1].append(bucket[j])
                continue
            dynamic_bucket[smallest_paths_grades[j][i]].append(bucket[j])
        bucket_length = [len(x) for x in dynamic_bucket]
    return bucket








