'''
* Team Id : GG_1110
* Author List : Aishini Bhattacharjee, Adithya S Ubaradka, Deepak C Nayak, Upasana Nayak
* Filename: shortest_path.py
* Theme: GeoGuide
* Classes: Node:
           Functions: add_neighbor
           Edge
           Graph:
           Functions: add_node, add_edge, delete_edge, get_nodes, get_edges
* Other Functions: find_3_node_lists, dijkstra, find_shortest_path, encoding
* Global Variables: cost_90, cost_180, se_90, reversed, set_90_right, set_90_left
'''
####################### IMPORT MODULES #######################
from itertools import permutations
import json
import heapq

cost_90 = 10 #cost for a 90 degree turn because it's more time-consuming than if it goes straight

#set_90 contains all 90 degree junctions on the arena
set_90 = [('Start_End', 'A', 'H'), ('A', 'B', "I"), ('C', 'B', 'I'), ('B', 'C', 'J'), ('D', 'C', 'J'), ('C', 'D', 'K'), ('K', 'E', 'F'), ('D', 'E', 'K'),
          ('E', 'F', 'J'), ('G', 'F', 'J'), ('F', 'G', 'I'), ('A', 'H', 'I'), ('B', 'I', 'J'), ('G', 'I', 'J'), ('B', 'I', 'H'), ('G', 'I', 'H'), ('H', 'A', 'B'),('G', 'H', 'I'),
          ('C', 'J', 'I'), ('F', 'J', 'I'), ('C', 'J', 'K'), ('F', 'J', 'K'), ('D', 'K', 'J'), ('E', 'K', 'J'), ('K', 'D', 'E')]
reversed = [(tup[2], tup[1], tup[0]) for tup in set_90] #To include both right and left turns, all tuples in the set were reversed and added
set_90.append(reversed)

#set_90_right contains right turns and set_90_left contains left turns
set_90_right = [('Start_End', 'A', 'H'), ('A', 'B', 'I'), ('I', 'B', 'C'), ('B', 'C', 'J'), ('J', 'C', 'D'), ('C', 'D', 'K'), ('K', 'E', 'F'), ('D', 'E', 'K'),
          ('E', 'F', 'J'), ('H', 'A', 'B'), ('J', 'F', 'G'), ('F', 'G', 'I'), ('I', 'G', 'H'), ('I', 'H', 'A'), ('J', 'I', 'B'),
          ('G', 'I', 'J'), ('B', 'I', 'H'), ('H', 'I', 'G'), ('G', 'H', 'I'),
          ('C', 'J', 'I'), ('I', 'J', 'F'), ('K', 'J', 'C'), ('F', 'J', 'K'), ('D', 'K', 'J'), ('J', 'K', 'E'), ('K', 'D', 'E')]
set_90_left = [(tup[2], tup[1], tup[0]) for tup in set_90_right]
##############################################################

'''Class Name: Node
Description: Represents a node in a graph with a name and a list of neighbors.'''
class Node:
    def __init__(self, name):
        self.name = name
        self.neighbors = []  # Dictionary to store neighbors and edge weights

    '''' Function Name: add_neighbor
     Input: neighbor (Node) - The neighbor node to be added to the current node's list of neighbors.
     Output: None
     Logic: Appends the provided neighbor to the list of neighbors of the current node.
     Example Call:
       node_A = Node("A")
       node_B = Node("B")
       node_A.add_neighbor(node_B)'''
    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)

''' Class Name: Edge
 Description: Represents an edge in a graph with source, destination, weight, and an attribute for event presence.
 Input: 
   - src (Node): The source node of the edge.
   - dst (Node): The destination node of the edge.
   - weight (int or float): The weight of the edge.
   - has_event (bool): A boolean attribute indicating the presence of an event on the edge.'''
class Edge:
    def __init__(self, src, dst, weight, has_event):
        self.src = src
        self.dst = dst
        self.weight = weight
        self.has_event = has_event  # Attribute for event presence

'''Class Name: Graph
Description: Represents a graph with nodes and edges.'''
class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    ''' Function Name: add_node
     Input: node (Node) - The node to be added to the graph.
     Output: None
     Logic: Adds the provided node to the graph's nodes dictionary.
     Example Call:
       graph = Graph()
       node_A = Node("A")
       graph.add_node(node_A)'''
    def add_node(self, node):
        self.nodes[node.name] = node

    ''' Function Name: add_edge
     Input:
       - src (Node): The source node of the edge.
       - dst (Node): The destination node of the edge.
       - weight (int or float): The weight of the edge.
       - has_event (bool): A boolean attribute indicating the presence of an event on the edge (default is False).
     Output: None
     Logic: Adds an edge to the graph, creating nodes if they don't exist and updating the graph's edges dictionary.
     Example Call:
       graph = Graph()
       node_A = Node("A")
       node_B = Node("B")
       graph.add_edge(node_A, node_B, 10, True)'''
    # Check if source node is not in the graph, add it
    def add_edge(self, src, dst, weight, has_event=False):
        # Check if source node is not in the graph, add it
        if src.name not in self.nodes:
            self.add_node(src)

        # Check if destination node is not in the graph, add it
        if dst.name not in self.nodes:
            self.add_node(dst)

        # Create directed edge from src to dst
        edge_created = Edge(src, dst, weight, has_event)
        # Create directed edge from dst to src
        edge_reversed = Edge(dst, src, weight, has_event)

        # Update neighbors for both source and destination nodes
        src.add_neighbor(dst)
        dst.add_neighbor(src)

        # Store edges in the edges dictionary with appropriate keys
        edge_key = (src.name, dst.name)
        self.edges[edge_key] = edge_created

        reversed_edge_key = (dst.name, src.name)
        self.edges[reversed_edge_key] = edge_reversed

    ''' Function Name: delete_edge
     Input: src_name (str) - The name of the source node, dst_name (str) - The name of the destination node.
     Output: None
     Logic: Deletes the edge between the source and destination nodes, if it exists.
     Example Call:
       graph = Graph()
       node_A = Node("A")
       node_B = Node("B")
       graph.add_edge(node_A, node_B, 10, True)
       graph.delete_edge("A", "B")'''
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

    ''' Function Name: get_nodes
     Input: None
     Output: Dictionary - A dictionary containing nodes in the graph.
     Logic: Returns the nodes dictionary of the graph.
     Example Call:
       graph = Graph()
       nodes_dict = graph.get_nodes()'''
    def get_nodes(self):
        return self.nodes

    ''' Function Name: get_edges
     Input: None
     Output: Dictionary - A dictionary containing edges in the graph.
     Logic: Returns the edges dictionary of the graph.
     Example Call:
       graph = Graph()
       edges_dict = graph.get_edges()'''
    def get_edges(self):
        return self.edges

'''Creating nodes for the arena in graphical representaion'''
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

'''Adding edges in the graph'''
graph.add_edge(node0, node1, 2) #Between start/end and the very first node that is visited
graph.add_edge(node1, node2, 5) #The edge upwards from node 1
graph.add_edge(node2, node3, 5) #The edge upwards from node 2
graph.add_edge(node3, node4, 4) #The edge upwards from node 3
graph.add_edge(node4, node5, 20, True) #A costly edge between nodes 4 and 5, because there are two rounded turns. This edge also has an associated event.
graph.add_edge(node5, node6, 4) #The edge downwards from node 5
graph.add_edge(node6, node7, 5) #The edge downwards from node 6
graph.add_edge(node7, node8, 11) #The bottom right edge taking a rounded turn from node 7, thus the added cost
graph.add_edge(node8, node9, 5) #The lowermost middle vertical edge
graph.add_edge(node9, node10, 5) #The middle section of the 3 vertical edges mid-arena
graph.add_edge(node10, node11, 4) #The uppermost middle vertical adge
graph.add_edge(node1, node8, 6, True) #Bottom left horizontal edge with an associated event
graph.add_edge(node2, node9, 6) #Second lowest horizontal edge on the left
graph.add_edge(node7, node9, 6, True) #Lowest straight horizontal edge on the right with an associated event
graph.add_edge(node3, node10, 6, True) #Third lowest horizontal edge on the left with an associated event
graph.add_edge(node4, node11, 6) #Topmost small horizontal edge on the left with an associated event
graph.add_edge(node6, node10, 6, True) #Second lowest horizontal edge on the right with event
graph.add_edge(node5, node11, 6) #Topmost small horizontal edge on the right

'''
 Function Name: find_3_node_lists
 Input: graph (Graph) - The graph for which 3-node lists need to be found.
 Output: List - A list of tuples containing unique 3-node lists.
 Logic: Finds all unique 3-node lists in the graph, considering nodes with at least 2 neighbors. Each such element in the list
        is a junction in the graph.
 Example call: result = find_3_node_lists(graph)'''
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

'''
 Function Name: dijkstra
 Input:
   - graph (Graph): The graph on which Dijkstra's algorithm will be applied.
   - start (str): The starting node for the algorithm.
   - end (str): The target node to find the shortest path to.
   - set_90 (set): The set of three-node junctions for identifying 90 degree turns.
 Output: List - The path taking up the least amount of time, from start to end.
 Logic: Implements a modified Dijkstra's algorithm to find the shortest path in a graph, in a way that 90 degree turns have
        an extra cost added to them, because it takes more time to make a turn than to go straight.
 Example call: path = dijkstra(graph, start, end, set_90)'''
def dijkstra(graph, start, end, set_90=set_90):
    # Initialize distances for all nodes to be infinite and sets predecessor to None for all
    distances = {node: float('infinity') for node in graph.get_nodes()}
    predecessors = {node: None for node in graph.get_nodes()}
    distances[start] = 0

    # Priority queue for Dijkstra's algorithm, initially containing only the starting node
    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        # Greater distances as compared to the ones previously assigned are rejected
        if current_distance > distances[current_node]:
            continue

        # Calculate distance of the current node from its neighbors and find out which is the shortest
        for neighbor in graph.nodes[current_node].neighbors:
            weight = graph.edges[(current_node, neighbor.name)].weight
            new_distance = distances[current_node] + weight

            # Update distances and predecessors if a shorter path is found, and push the same in the priority queue
            if new_distance < distances[neighbor.name]:
                distances[neighbor.name] = new_distance
                predecessors[neighbor.name] = current_node
                heapq.heappush(priority_queue, (new_distance, neighbor.name))

                # Check if the current three-node junction is in set_90 and add the additional cost
                if predecessors[current_node] is not None:
                    current_path = [current_node, predecessors[current_node], neighbor.name]
                    # The junction consists of the predecessor, the current node, and the neighbor of the current node
                    # in that order.
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

''' Function Name: find_shortest_path
 Input:
   - graph (Graph): The graph for which the shortest path needs to be found.
   - priorities (list): A list of tuples representing edge priorities.
 Output: List - The shortest path considering priorities as well as turning costs.
 Logic: Finds the least time-consuming path in the graph in the following way:
    Algorithm:
    1. Initialize the starting and ending points for the path.
    2. Create an initial path starting from the start point.
    3. Iterate through the list of node pairs depicting edges that have events (priorities):
        a. Find the shortest paths from the starting point to each node in the current pair.
        b. Choose the shorter path between the forward and backward paths.
        c. Optimize the path by potentially deleting edges in case a 180 degree turn is encountered, because that
        would place a really high cost on the travel time.
    4. If it's the last priority edge, find the shortest path back to the ending point.
    5. Return the total optimized path through the graph.
 Example call: path = find_shortest_path(graph, priorities)'''

def find_shortest_path(graph, priorities):
    start = "Start_End"

    # Initialisation of forward path
    total_path_forward = [start]

    for i, (node1, node2) in enumerate(priorities):
        # Find the shortest path to the edge with the current priority
        shortest_path_forward = dijkstra(graph, start, node1) + [node2]
        shortest_path_backward = dijkstra(graph, start, node2) + [node1]
        # Amongst shortest path forward and shortest path backward, the shorter of the two is chosen as indicated below,
        # to consider the nearer end-point of the priority edge.

        # Choose the shorter path between forward and backward
        if len(shortest_path_backward) < len(shortest_path_forward):
            # The following condition checks is there are atleast 2 nodes in the path, this is important
            # because we will be checking if the path has a 180 degree turn in the next if statement and
            # that assumes a minimum of 2 nodes in the already existing path.
            if len(total_path_forward) >= 2:
                # Here, we check if there is a 180 degree turn by checking if the last second node is the
                # same as the node we are about to add to the path. (check for [A-->B-->A])
                # If there is a 180 degree turn, that particular edge will be deleted since it is an
                # expensive maneuver and the dijikstra algorithm will be run again without that
                # edge to find an alternative path.
                if total_path_forward[-2] == shortest_path_forward[1]:
                    # graph_temp is the newly constructed graph, without the deleted edge.
                    graph_temp = graph.delete_edge(total_path_forward[-2], total_path_forward[-1])
                    # We re-run the dijikstra algorithm to find the shortest paths in thr new graph
                    shortest_path_forward = dijkstra(graph_temp, start, node1) + [node2]
                    shortest_path_backward = dijkstra(graph_temp, start, node2) + [node1]
                    # The node which is nearer to the current location of the bot is considered.
                    if len(shortest_path_backward) < len(shortest_path_forward):
                        total_path_forward += shortest_path_backward[1:]
                        # the start node is set to the ending node of this current priority edge
                        start = total_path_forward[-1]
                    else:
                        total_path_forward += shortest_path_forward[1:]
                        start = total_path_forward[-1]
                else:
                    # if there was no 180 degree turn, we simply add the found path to the existing path.
                    total_path_forward += shortest_path_forward[1:]
                    start = total_path_forward[-1]
            else:
                total_path_forward += shortest_path_backward[1:]
                start = total_path_forward[-1]
        else:
            # Similar checks and handling for forward path not being optimal
            if len(total_path_forward) >= 2:
                if total_path_forward[-2] == shortest_path_backward[1]:
                    #edge delete
                    graph_temp = graph.delete_edge(total_path_forward[-2], total_path_forward[-1])
                    shortest_path_forward = dijkstra(graph_temp, start, node1) + [node2]
                    shortest_path_backward = dijkstra(graph_temp, start, node2) + [node1]
                    if len(shortest_path_backward) < len(shortest_path_forward):
                        total_path_forward += shortest_path_backward[1:]  # Exclude start node of the path
                        start = total_path_forward[-1]
                    else:
                        total_path_forward += shortest_path_forward[1:]
                        start = total_path_forward[-1]
                else:
                    total_path_forward += shortest_path_backward[1:]
                    start = total_path_forward[-1]
            else:
                total_path_forward += shortest_path_forward[1:]
                start = total_path_forward[-1]
        # When the iterable i reaches the value len(priorities) - 1, it indicates that all the edges containing
        # events have been traversed. From the ending node of the last visited edge, the shortest path backwards
        # to the starting point is calculated.
        if i == len(priorities) - 1:
            end = "Start_End"
            last_node = total_path_forward[-1]
            print(last_node)
            # Shortest path from the last node to start/end
            shortest_path_backward = dijkstra(graph, last_node, end)
            if total_path_forward[-2] == shortest_path_backward[1]:
                # Removal of the edge in the case of a 180 degree turn
                graph_temp = graph.delete_edge(total_path_forward[-2], total_path_forward[-1])
                shortest_path_backward = dijkstra(graph_temp, last_node, end)
            total_path_forward += shortest_path_backward[1:]
    return total_path_forward

with open('priority_edge_order.json', 'r') as json_file:
    loaded_m = json.load(json_file)
priorities = loaded_m
shortest_path = find_shortest_path(graph, priorities)
print(shortest_path)

''' Function Name: encoding
     Input: shortest_path : returned shortest path by the find_shortest_path function
     Output: encoded path as a list of directions
     Logic: All consecutive 3 nodes, which form junctions in the graph, are considered iteratively. Consulting set_90_right 
            and set_90_left lists, we get the junctions in which right turns and left turns are involved. Junctions
            which are not part of either list do not have any turns. Right turns are encoded as '2', left turns as '1'
            and straight junctions as '0'.
     Example Call: path = encoding(shortest_path)'''
def encoding(shortest_path = shortest_path):
    encoding = []
    for i in range(len(shortest_path) - 2):
        sub_sequence = tuple(shortest_path[i:i+3]) # Considering 3 consecutive elements in the sequence for a junction
        if sub_sequence in set_90_right:
            encoding.append(2)
        elif sub_sequence in set_90_left:
            encoding.append(1)
        else:
            encoding.append(0)
        return encoding

encoded_path = encoding(shortest_path)
print(encoded_path)

# The encoded_path is saved as a json file and is later retrieved in the communication.py file.
with open('encoded_path.json', 'w') as json_file:
    json.dump(encoding, json_file)
