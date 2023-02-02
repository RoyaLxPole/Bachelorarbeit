import numpy as np
import time
import graph_generator as gg
from Graph import Graph
from tqdm import tqdm
import large_random_graph_generator as lrgg


import help
#WIN = pygame.display.set_mode((WIDTH, HEIGHT))

couldnsovlematrix = np.array([  [6, 5, 4, 3, 3, 3, 3, 0],
                                [6, 5, 4, 3, 3, 0, 0, 3],
                                [6, 5, 4, 0, 0, 3, 0, 3],
                                [6, 5, 0, 3, 0, 3, 0, 0],
                                [6, 5, 0, 0, 3, 0, 3, 0],
                                [6, 0, 4, 3, 0, 3, 0, 0],
                                [6, 0, 0, 0, 3, 0, 3, 3],
                                [0, 5, 4, 0, 0, 0, 3, 3]])

impsolvematrix = np.array([[3, 3, 3, 3],
                          [3, 3, 3, 3],
                          [3, 3, 3, 3],
                          [3, 3, 3, 3]])

matrix = np.array([ [2,4,3,0,0],
                    [2,4,3,2,1],
                    [2,4,3,2,0],
                    [0,4,3,2,0],
                    [0,4,0,0,1]])

matrix1 = np.array([[2,2,4,0,0,0,0,0,0,0,0,0],
                    [2,2,0,4,0,0,0,0,0,0,0,0],
                    [2,0,4,4,0,0,3,4,0,0,0,0],
                    [0,2,4,4,3,3,0,0,0,0,0,0],
                    [0,0,0,4,3,3,0,0,0,0,2,0],
                    [0,0,0,4,3,3,3,0,0,0,0,0],
                    [0,0,4,0,0,3,3,4,0,0,0,0],
                    [0,0,4,0,0,0,3,4,3,0,0,2],
                    [0,0,0,0,0,0,0,4,3,2,0,2],
                    [0,0,0,0,0,0,0,0,3,2,2,0],
                    [0,0,0,0,3,0,0,0,0,2,2,0],
                    [0,0,0,0,0,0,0,4,3,0,0,2]])

matrix2 = np.array([[3,2,0,0,0,0,0,0,5,3,0,0,0,0,0,0],
                    [3,2,4,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,2,4,2,0,0,0,3,5,0,0,0,0,0,0,0],
                    [0,0,4,2,2,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,2,2,2,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,2,2,4,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,2,4,3,0,0,0,4,3,0,0,0],
                    [0,0,4,0,0,0,4,3,5,0,0,0,0,0,0,0],
                    [3,0,4,0,0,0,0,3,5,3,4,0,0,0,0,0],
                    [3,0,0,0,0,0,0,0,5,3,4,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,5,3,4,4,0,0,0,0],
                    [0,0,0,0,0,0,4,0,0,0,4,4,3,3,0,0],
                    [0,0,0,0,0,0,4,0,0,0,0,4,3,3,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,4,3,3,0,2],
                    [0,0,0,0,0,0,0,0,0,0,4,0,0,0,2,2],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,3,2,2]])

matrix3 = np.array([[3,1,1,1,0,0], #impossible to solve because not connected
                   [3,1,0,0,0,0],
                   [3,0,1,0,0,0],
                   [3,0,0,1,0,0],
                   [0,0,0,0,1,1],
                   [0,0,0,0,1,1]])
kette = np.array([[1,2,0,0,0],
                  [1,2,2,0,0],
                  [0,2,2,2,0],
                  [0,0,2,2,1],
                  [0,0,0,2,1]])

matrix4 = np.array([[3,3,5,0,0,3,0,0,0,0,0,0,0],
                    [3,3,5,0,3,0,0,0,0,0,0,0,0],
                    [3,3,5,4,3,3,0,0,0,0,0,0,0],
                    [0,0,5,4,3,3,2,0,0,0,0,0,0],
                    [0,3,5,4,3,0,0,0,0,0,0,0,0],
                    [3,0,5,4,0,3,0,0,0,0,0,0,0],
                    [0,0,0,4,0,0,2,4,0,0,0,0,0],
                    [0,0,0,0,0,0,2,4,3,5,0,3,0],
                    [0,0,0,0,0,0,0,4,3,5,3,0,0],
                    [0,0,0,0,0,0,0,4,3,5,3,3,3],
                    [0,0,0,0,0,0,0,0,3,5,3,0,3],
                    [0,0,0,0,0,0,0,4,0,5,0,3,3],
                    [0,0,0,0,0,0,0,0,0,5,3,3,3]])

                    #0 1 2 3 4 5 6 7 8 9 1011121314
matrix6 = np.array([[3,3,3,6,0,0,0,0,0,0,0,0,0,0,0],#0nicht lösbar 0er als antwort
                    [3,3,0,6,0,0,3,0,0,0,0,0,0,0,0],#1
                    [3,0,3,6,0,3,0,0,0,0,0,0,0,0,0],#2
                    [3,3,3,6,4,3,3,0,0,0,0,0,0,0,0],#3
                    [0,0,0,6,4,3,3,2,0,0,0,0,0,0,0],#4
                    [0,0,3,6,4,3,0,0,0,0,0,0,0,0,0],#5
                    [0,3,0,6,4,0,3,0,0,0,0,0,0,0,0],#6
                    [0,0,0,0,4,0,0,2,4,0,0,0,0,0,0],#7
                    [0,0,0,0,0,0,0,2,4,3,6,0,3,0,0],#8
                    [0,0,0,0,0,0,0,0,4,3,6,3,0,0,0],#9
                    [0,0,0,0,0,0,0,0,4,3,6,3,3,3,3],#10
                    [0,0,0,0,0,0,0,0,0,3,6,3,0,3,0],#11
                    [0,0,0,0,0,0,0,0,4,0,6,0,3,0,3],#12
                    [0,0,0,0,0,0,0,0,0,0,6,3,0,3,3],#13
                    [0,0,0,0,0,0,0,0,0,0,6,0,3,3,3]])#14

matrix5 = np.array([[3,3,5,0,0,3,0,0,0,0,0,0], #impossible to solve because not connected
                    [3,3,5,0,3,0,0,0,0,0,0,0],
                    [3,3,5,4,3,3,0,0,0,0,0,0],
                    [0,0,5,4,3,3,0,0,0,0,0,0],
                    [0,3,5,4,3,0,0,0,0,0,0,0],
                    [3,0,5,4,0,3,0,0,0,0,0,0],
                    [0,0,0,0,0,0,4,3,5,0,3,0],
                    [0,0,0,0,0,0,4,3,5,3,0,0],
                    [0,0,0,0,0,0,4,3,5,3,3,3],
                    [0,0,0,0,0,0,0,3,5,3,0,3],
                    [0,0,0,0,0,0,4,0,5,0,3,3],
                    [0,0,0,0,0,0,0,0,5,3,3,3]])

impossiblematrix = np.array([[4,4,4,3,0,0,0,0,0,4],
                            [4,4,4,0,4,0,0,0,3,0],
                            [4,4,4,0,0,4,0,3,0,0],
                            [4,0,0,3,4,4,0,0,0,0],
                            [0,4,0,3,4,0,3,3,0,0],
                            [0,0,4,3,0,4,3,0,3,0],
                            [0,0,0,0,4,4,3,0,0,4],
                            [0,0,4,0,4,0,0,3,0,4],
                            [0,4,0,0,0,4,0,0,3,4],
                            [4,0,0,0,0,0,3,3,3,4]])

difficultmatrix = np.array([
                            [3,4,0,4,0,0,0,0,0,0,0,0,0,0,3,0],
                            [3,4,3,0,3,0,4,0,0,0,0,0,0,0,0,0],
                            [0,4,3,4,0,0,0,0,0,0,0,0,0,0,0,3],
                            [3,0,3,4,0,3,0,4,0,0,0,0,0,0,0,0],
                            [0,4,0,0,3,0,0,4,0,0,3,0,0,0,0,0],
                            [0,0,0,4,3,0,4,0,0,0,0,0,4,0,0,0],
                            [0,4,0,0,0,3,4,0,4,4,0,0,0,0,0,0],
                            [0,0,0,4,3,0,0,4,4,4,0,0,0,0,0,0],
                            [0,0,0,0,0,0,4,4,4,0,3,0,4,0,0,0],
                            [0,0,0,0,0,0,4,4,0,4,0,3,0,3,0,0],
                            [0,0,0,0,3,0,0,0,4,0,3,3,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,3,3,4,0,3,0],
                            [0,0,0,0,0,3,0,0,4,0,0,3,4,3,0,0],
                            [0,0,0,0,0,0,0,0,0,4,0,0,4,3,0,3],
                            [3,0,0,0,0,0,0,0,0,0,0,3,0,0,3,3],
                            [0,0,3,0,0,0,0,0,0,0,0,0,0,3,3,3]])



'''def main():
    #backtracking und meinen algorithmus nicht gleichzeitig laufen lassen das ergebniss von backtracking wird verfälscht
    gg.graph_generator(matrix)
    coloring = [''] * len(matrix2)
    pos_coloring = []
    for i in range(len(matrix2)):
        pos_coloring.append(['r', 'b', 'g'])
    #help.run_sort_matrix(matrix2)
    print(help.backtracking(matrix2, set(), coloring, pos_coloring, 0))'''

def main():

    #help.run_sort_matrix(matrix2)





    '''index = 9
    matrix_collection, possible_coloring = gg.read_matrix_from_file('n={}.txt'.format(index), index)

    for i in tqdm(range(len(matrix_collection))):
        #print(matrix_collection[i])
        gg.graph_generator(matrix_collection[i], index + 1)'''
    '''    prozent = 70
    n = 150
    lrgg.generate_random_graph(n,1/(n - (n//100 * prozent)), 100000000)
'''

    visited = set()
    index = 0
    matrix = difficultmatrix
    yes_no, pos_coloring = gg.backtracking(matrix)
    print('Is coroble:{}'.format(yes_no))
    print('coloring')
    print(pos_coloring)
    colors, _, _ = help.run_sort_matrix(matrix)
    print(colors)
    #gg.sort_matrix_files('Graphen\\Graphen50\\n=50_right_format.txt')

    '''file = open('times1', 'a')
    sizes = [8,9, 10]

    for size in sizes:
        st = time.process_time()
        coloring_time = gg.read_matrix_and_run_my_algorithm('Graphen\\n={}.txt'.format(size), size)
        et = time.process_time()
        res1 = et - st
        ''''''st = time.process_time()
        gg.read_file_and_do_backtracking('Graphe\\n={}.txt'.format(size), size)
        et = time.process_time()
        res2 = et - st''''''
        #size = 8
        #help.check_file_for_coloring('n={}.txt'.format(size), size)
        file.write('{}\n'.format(size))
        file.write('{}\n'.format(res1))
        file.write('{}\n'.format(coloring_time))
        #file.write('{}\n'.format(res2))
        file.write('\n')
        file.write('\n')'''

if __name__ == "__main__":
    main()



'''def draw_window(win):
    win.fill(BG_COLOR)
    pygame.display.update()


def draw_node(graph, number):
    pos = pygame.mouse.get_pos()
    circle_pos = pygame.draw.circle(win, BLACK, pos, 20, 3)
    font = get_font(10)
    text_surface = font.render(str(number), 1, BLACK)
    win.blit(text_surface, pos)
    graph.add_node(circle_pos, number)
    print(graph.nodes)


def main2():
    global win, last_pos
    win = WIN
    graph = Graph()
    pygame.display.set_caption("3 Coloring Simulation")
    drawing_edge = False
    run = True
    clock = pygame.time.Clock()
    draw_window(WIN)
    counter = 0
    number = 0
    while run:
        clock.tick(2)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if pygame.mouse.get_pressed()[0]:
                print('pressed')
                draw_node(graph, number)
                number = number + 1
                if counter % 2 == 0 and counter > 0 and drawing_edge == True:
                    current_pos = pygame.mouse.get_pos()
                    pygame.draw.line(win, BLACK, last_pos, current_pos, 3)
                last_pos = pygame.mouse.get_pos()
                counter = counter + 1
        pygame.display.update()

    pygame.quit()'''


