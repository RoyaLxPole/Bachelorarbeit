
class Graph:
    def __init__(self):
        self.nodes = []
        self.matrix = None
        self.connections = []

    def add_node(self, node_pos, number):
        self.nodes.append((node_pos, number))

    def add_connection(self, node1, node2):
        self.connections.append((node1, node2))


