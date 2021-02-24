import numpy as np
from numpy.linalg import norm

class Graph_Node():
    def __init__(self, data=[], label=None):
        #data = [original data, stretched data, squeezed data]
        self.data = data
        self.label = label
        self.neighbors = [] #[distance, node]

    #Return the unit vector of the vector
    def unit_vector(self, vector):
        return vector / np.linalg.norm(vector)

    #Return the angle in radians between two vectors
    def angle_between(self, v1, v2):
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    #takes angle between two nodes' vectors and inverse-vectors and multiplies them together
    def node_distance(self, node):
        sp = self.angle_between(self.data[1], node.data[1])
        sq = self.angle_between(self.data[2], node.data[2])
        return sp*sq

    #Be careful not to add same neighbor more than once
    def add_neighbor(self, node, distance):
        self.neighbors.append([distance, node])

    def compare_neighbors(self, node, visited_nodes, travel_path=None, parent=None, distance_so_far=None):

        distance = self.node_distance(node)
        fixed_node = self
        steps = 1
        node_similarity_score = 0
        neighbor_similarity_score = None
        num_neighbors = len(self.neighbors)
        min_distance = distance
        min_travel_path = []
        branch_path = None
        
        dsf = 0

        if distance_so_far is not None:
            dsf = distance_so_far + distance

        if travel_path is not None:
            travel_path.append([self.label, distance, dsf])
        
        min_travel_path.append([self.label, distance, dsf])

        if num_neighbors > 0:
            
            for dist_from_self, neighbor in self.neighbors:

                #if parent is not None and neighbor.label == parent.label:
                if neighbor.label in visited_nodes:
                    dist = neighbor.node_distance(node)
                    node_similarity_score = abs(dist - dist_from_self)
                    continue

                visited_nodes.add(self.label)

  
                #TODO: reduce total number of steps taken via machine learning?
                n_node, n_dist, n_steps, n_similarity_score, n_neighbors, n_visited_nodes, n_travel_path, n_branch = neighbor.compare_neighbors(node, visited_nodes=visited_nodes, travel_path=travel_path, parent=self, distance_so_far=dsf)
                steps += n_steps   #number of total steps taken so far

                if n_dist < min_distance:
                    min_distance = n_dist
                    fixed_node = n_node
                    branch_path = n_branch
                    node_similarity_score = n_similarity_score
                    num_neighbors = n_neighbors
        else:
            min_distance = distance
            fixed_node = self
            branch_path = None
            node_similarity_score = distance
            num_neighbors = 0

        if branch_path is not None:
            min_travel_path.extend(branch_path)

        return fixed_node, min_distance, steps, node_similarity_score, num_neighbors, visited_nodes, travel_path, min_travel_path
