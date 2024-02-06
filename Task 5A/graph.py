complex_turn_cost = 20

class Node:
    def __init__(self, name):
        self.name = name
        self.neighbors = {}  # Dictionary to store neighbors and edge weights

    def add_neighbor(self, neighbor, weight):
        self.neighbors[neighbor] = weight

class Edge:
    def __init__(self, src, dst, weight, has_event, has_complex_turn):
        self.src = src
        self.dst = dst
        self.weight = weight
        self.has_event = has_event  # Attribute for event presence
        self.has_complex_turn = has_complex_turn  # Attribute for complex turn

class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node(self, node):
        self.nodes[node.name] = node

    def add_edge(self, src, dst, weight, has_event=False, has_complex_turn=False):
        if src not in self.nodes:
            self.add_node(src)
        if dst not in self.nodes:
            self.add_node(dst)
        edge_created = Edge(src, dst, weight, has_event, has_complex_turn)
        if edge_created.has_complex_turn:
            edge_created.weight = edge_created.weight + complex_turn_cost
        src.add_neighbor(dst, weight)
        self.edges.append(edge_created)


    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges

# Example usage
graph = Graph()
node0 = Node("Start_End")
node1 = Node("A")
node2 = Node("B")
node3 = Node("C")
node4 = Node("D")
node5 = Node("E")
node6 = Node("F")
node7 = Node("G")
node8 = Node("H")
node9 = Node("I")
node10 = Node("J")
node11= Node("K")

graph.add_edge(node0, node1, 2)
graph.add_edge(node1, node2, 5)
graph.add_edge(node2, node3, 5)
graph.add_edge(node3, node4, 4)
graph.add_edge(node4, node5, 16, True, True)
graph.add_edge(node5, node6, 4)
graph.add_edge(node6, node7, 5)
graph.add_edge(node7, node8, 10, False, True)
graph.add_edge(node8, node9, 5)
graph.add_edge(node9, node10, 5)
graph.add_edge(node10, node11, 4)
graph.add_edge(node1, node8, 5, True, False)
graph.add_edge(node2, node9, 5)
graph.add_edge(node7, node9, 5, True, False)
graph.add_edge(node3, node10, 5, True, False)
graph.add_edge(node4, node11, 5)
graph.add_edge(node6, node10, 5, True, False)
graph.add_edge(node5, node11, 5)

print(graph.nodes[0].neighbors)

# Accessing edge attributes
edge1 = graph.edges[0]
print(edge1.has_event)  # True
print(edge1.has_complex_turn)  # False
