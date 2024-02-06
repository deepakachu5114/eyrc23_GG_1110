'''
*****************************************************************************************
*
*        		===============================================
*           		Geo Guide (GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement the finding of the shortest path for Task 5A of Geo Guide (GG) Theme (eYRC 2023-24).
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:			GG_1110
# Author List:		Aishini Bhattacharjee, Adithya S Ubaradka, Deepak C Nayak, Upasana Nayak
# Filename:			shortest_path.py


####################### IMPORT MODULES #######################
import heapq
import json
cost_90 = 10
cost_180 = 10
##############################################################

class Node:
    def __init__(self, name):
        self.name = name
        self.neighbors = []  # Dictionary to store neighbors and edge weights

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)

class Edge:
    def __init__(self, src, dst, weight, has_event):
        self.src = src
        self.dst = dst
        self.weight = weight
        self.has_event = has_event  # Attribute for event presence

class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def add_node(self, node):
        self.nodes[node.name] = node

    def add_edge(self, src, dst, weight, has_event=False):
        if src.name not in self.nodes:
            self.add_node(src)
        if dst.name not in self.nodes:
            self.add_node(dst)
        edge_created = Edge(src, dst, weight, has_event)
        edge_reversed = Edge(dst, src, weight, has_event)

        src.add_neighbor(dst)
        dst.add_neighbor(src)

        edge_key = (src.name, dst.name)
        self.edges[edge_key] = edge_created
        reversed_edge_key = (dst.name, src.name)
        self.edges[reversed_edge_key] = edge_reversed

    def delete_edge(self, src_name, dst_name):
        if src_name not in self.nodes or dst_name not in self.nodes:
            return  # Edge or nodes do not exist

        src = self.nodes[src_name]
        dst = self.nodes[dst_name]

        edge_key = (src.name, dst.name)
        reversed_edge_key = (dst.name, src.name)

        if edge_key in self.edges:
            del self.edges[edge_key]

        if reversed_edge_key in self.edges:
            del self.edges[reversed_edge_key]

        # Remove the edge from the neighbors list
        if dst in src.neighbors:
            src.neighbors.remove(dst)

        if src in dst.neighbors:
            dst.neighbors.remove(src)
        return self

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
graph.add_edge(node4, node5, 31, True)
graph.add_edge(node5, node6, 4)
graph.add_edge(node6, node7, 5)
graph.add_edge(node7, node8, 18)
graph.add_edge(node8, node9, 5)
graph.add_edge(node9, node10, 5)
graph.add_edge(node10, node11, 4)
graph.add_edge(node1, node8, 5, True)
graph.add_edge(node2, node9, 5)
graph.add_edge(node7, node9, 5, True)
graph.add_edge(node3, node10, 5, True)
graph.add_edge(node4, node11, 5)
graph.add_edge(node6, node10, 5, True)
graph.add_edge(node5, node11, 5)

print([neighbor.name for neighbor in graph.nodes["I"].neighbors])

# Accessing edge attributes
edge1 = graph.edges[('F', 'G')]
print(edge1.has_event)  # True

from itertools import permutations

def find_3_node_lists(graph):
    three_node_lists = set()

    for node in graph.nodes.values():
        if len(node.neighbors) >= 2:
            # Generate all 3-permutations with the node as the middle element
            for perm in permutations(node.neighbors, 2):
                three_node_list = [perm[0].name, node.name, perm[1].name]
                # Check if all elements are distinct before adding to the set
                if len(set(three_node_list)) == len(three_node_list):
                    three_node_lists.add(tuple(three_node_list))

    return list(three_node_lists)

# Example usage
result = find_3_node_lists(graph)
print(result)
print(len(result))

set_90 = [('Start_End', 'A', 'H'), ('A', 'B', "I"), ('C', 'B', 'I'), ('B', 'C', 'J'), ('D', 'C', 'J'), ('C', 'D', 'K'), ('K', 'E', 'F'), ('D', 'E', 'K'),
          ('E', 'F', 'J'), ('G', 'F', 'J'), ('F', 'G', 'I'), ('A', 'H', 'I'), ('B', 'I', 'J'), ('G', 'I', 'J'), ('B', 'I', 'H'), ('G', 'I', 'H'), ('H', 'A', 'B'),('G', 'H', 'I'),
          ('C', 'J', 'I'), ('F', 'J', 'I'), ('C', 'J', 'K'), ('F', 'J', 'K'), ('D', 'K', 'J'), ('E', 'K', 'J'), ('K', 'D', 'E')]
reversed = [(tup[2], tup[1], tup[0]) for tup in set_90]
set_90.append(reversed)


set_90_right = [('Start_End', 'A', 'H'), ('A', 'B', 'I'), ('I', 'B', 'C'), ('B', 'C', 'J'), ('J', 'C', 'D'), ('C', 'D', 'K'), ('K', 'E', 'F'), ('D', 'E', 'K'),
          ('E', 'F', 'J'), ('H', 'A', 'B'), ('J', 'F', 'G'), ('F', 'G', 'I'), ('I', 'G', 'H'), ('I', 'H', 'A'), ('J', 'I', 'B'),
          ('G', 'I', 'J'), ('B', 'I', 'H'), ('H', 'I', 'G'), ('G', 'H', 'I'),
          ('C', 'J', 'I'), ('I', 'J', 'F'), ('K', 'J', 'C'), ('F', 'J', 'K'), ('D', 'K', 'J'), ('J', 'K', 'E'), ('K', 'D', 'E')]
set_90_left = [(tup[2], tup[1], tup[0]) for tup in set_90_right]

import heapq

# ... (Previous code remains the same)

def dijkstra(graph, start, end, set_90=set_90):
    # Initialize distances and predecessor
    distances = {node: float('infinity') for node in graph.get_nodes()}
    predecessors = {node: None for node in graph.get_nodes()}
    distances[start] = 0

    # Priority queue for Dijkstra's algorithm
    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        # Check if the current path to the node is shorter than the recorded distance
        if current_distance > distances[current_node]:
            continue

        for neighbor in graph.nodes[current_node].neighbors:
            weight = graph.edges[(current_node, neighbor.name)].weight
            new_distance = distances[current_node] + weight

            # Update distances and predecessors if a shorter path is found
            if new_distance < distances[neighbor.name]:
                distances[neighbor.name] = new_distance
                predecessors[neighbor.name] = current_node
                heapq.heappush(priority_queue, (new_distance, neighbor.name))

                # Check if the current three-node path is in set_90 and add the additional cost
                if predecessors[current_node] is not None:
                    current_path = [current_node, predecessors[current_node], neighbor.name]
                    current_path_tuple = tuple(sorted(current_path))
                    if current_path_tuple in set_90:
                        distances[neighbor.name] += cost_90

    # Reconstruct the path from end to start
    path = []
    current = end
    while current is not None:
        path.insert(0, current)
        current = predecessors[current]

    return path


def find_shortest_path(graph, priorities):
    start = "Start_End"
    end = "Start_End"

    total_path_forward = [start]

    for i, (node1, node2) in enumerate(priorities):
        # Find the shortest path to the edge with the current priority
        shortest_path_forward = dijkstra(graph, start, node1) + [node2]

        shortest_path_backward = dijkstra(graph, start, node2) + [node1]

        print("for",shortest_path_forward)
        print("back",shortest_path_backward)
        print("total", total_path_forward)

        # Choose the shorter path between forward and backward
        if len(shortest_path_backward) < len(shortest_path_forward):
            if len(total_path_forward) >= 2 and total_path_forward[-2] == shortest_path_backward[1]:
                print("back not optimal")
                if total_path_forward[-2] == shortest_path_forward[1]:
                    print("front also not optimal, edge deletion taking place")
                    graph_temp = graph.delete_edge(total_path_forward[-2], total_path_forward[-1])
                    print(f"del {total_path_forward[-2], total_path_forward[-1]}")
                    shortest_path_forward = dijkstra(graph_temp, start, node1) + [node2]
                    shortest_path_backward = dijkstra(graph_temp, start, node2) + [node1]
                    print(f"new paths {shortest_path_forward}\n{shortest_path_backward}")
                    if len(shortest_path_backward) < len(shortest_path_forward):
                        total_path_forward += shortest_path_backward[1:]  # Exclude start node of the path
                        start = total_path_forward[-1]
                    else:
                        total_path_forward += shortest_path_forward[1:]  # Exclude start node of the path
                        start = total_path_forward[-1]
                else:
                    total_path_forward += shortest_path_forward[1:]  # Exclude start node of the path
                    start = total_path_forward[-1]
            else:
                total_path_forward += shortest_path_backward[1:]  # Exclude start node of the path
                start = total_path_forward[-1]
        else:
            if len(total_path_forward) >= 2 and total_path_forward[-2] == shortest_path_forward[1]:
                print("front not optimal")
                if total_path_forward[-2] == shortest_path_backward[1]:
                    print("back also not optimal, deleting edge")
                    #edge delete
                    graph_temp = graph.delete_edge(total_path_forward[-2], total_path_forward[-1])
                    print(f"del {total_path_forward[-2], total_path_forward[-1]}")
                    shortest_path_forward = dijkstra(graph_temp, start, node1) + [node2]
                    shortest_path_backward = dijkstra(graph_temp, start, node2) + [node1]
                    print(f"new paths {shortest_path_forward}\n{shortest_path_backward}")
                    if len(shortest_path_backward) < len(shortest_path_forward):
                        total_path_forward += shortest_path_backward[1:]  # Exclude start node of the path
                        start = total_path_forward[-1]
                    else:
                        total_path_forward += shortest_path_forward[1:]  # Exclude start node of the path
                        start = total_path_forward[-1]
                else:
                    total_path_forward += shortest_path_backward[1:]  # Exclude start node of the path
                    start = total_path_forward[-1]
            else:
                total_path_forward += shortest_path_forward[1:]  # Exclude start node of the path
                start = total_path_forward[-1]
        # If it's the last priority, find the shortest path back to the ending point
        if i == len(priorities) - 1:
            end = "Start_End"
            last_node = total_path_forward[-1]
            print(last_node)
            shortest_path_backward = dijkstra(graph, last_node, end)
            if total_path_forward[-2] == shortest_path_backward[1]:
                print("optimising edge, deleting node")
                graph_temp = graph.delete_edge(total_path_forward[-2], total_path_forward[-1])
                print(f"del {total_path_forward[-2], total_path_forward[-1]}")
                shortest_path_backward = dijkstra(graph_temp, last_node, end)
                print(f"new path {shortest_path_backward}")
            total_path_forward += shortest_path_backward[1:]  # Exclude start node of the path
    return total_path_forward






# Example usage
with open('priority_edge_order.json', 'r') as json_file:
    loaded_m = json.load(json_file)
priorities = loaded_m
shortest_path = find_shortest_path(graph, priorities)
print(shortest_path)

encoding = []

for i in range(len(shortest_path) - 2):
    sub_sequence = tuple(shortest_path[i:i+3])
    if sub_sequence in set_90_right:
        encoding.append(2)
    elif sub_sequence in set_90_left:
        encoding.append(1)
    else:
        encoding.append(0)

print(encoding)

with open('encoded_path.json', 'w') as json_file:
    json.dump(encoding, json_file)
