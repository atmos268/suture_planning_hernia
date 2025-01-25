import math
import scipy.interpolate as inter
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import networkx as nx
from heapq import heappop, heappush
from itertools import count
from utils2 import euclidean_dist

class ForceModel:
    def __init__(self, mesh, ellipse_eccentricity: float, force_decay: float, neighbors_to_sample: int, dist_limit_factor = 2.0):
        # Initialize the ForceModel class
        self.ellipse_eccentricity = ellipse_eccentricity  # Eccentricity of the distance metric
        self.force_decay = force_decay # Rate at which the force decays with distance
        self.neighbors_to_sample = neighbors_to_sample # Neighbors to sample for the plane estimation
        self.mesh = mesh # Mesh of the wound, represented as a networkx graph
        self.dist_limit_factor = dist_limit_factor # Limit search distance between points for the shortest path

    def get_nearest_mesh_pt(self, point):
        # Get the nearest point on the mesh
        pass

    def euc_dist(self, point1, point2):
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + + (point1[2] - point2[2])**2)

    def get_position(self, point):
        return self.mesh.nodes[point]['pos']

    def get_shortest_path(self, point1, point2):

        # If the distance between the two points is greater than the distance limit, return None
        dist_limit = self.dist_limit_factor / self.force_decay

        if self.euc_dist(point1, point2) > dist_limit:
            return None

        mesh1 = self.get_nearest_mesh_pt(point1)
        mesh2 = self.get_nearest_mesh_pt(point2)

        c = count()
        queue = [(0, next(c), mesh1, 0, None)]


        # credit: astar.py in Networks package
        enqueued = {}
        explored = {}

        final_path = []
        final_len = 0

        while queue:
            priority, _,  popped_pt, curr_dist, parent = heappop(queue)

            # If the minimum possible distance is larger than the limit, return None
            if priority > dist_limit:
                return None

            if popped_pt == mesh2:
                path = [popped_pt]
                node = parent
                while node is not None:
                    path.append(node)
                    node = explored[node]
                path.reverse()
                final_path = path
                final_len = priority
                break

            if popped_pt in explored:
                # Do not override the parent of starting node
                if explored[popped_pt] is None:
                    continue

                # Skip bad paths that were enqueued before finding a better one
                qcost, heuristic = enqueued[popped_pt]
                if qcost < curr_dist:
                    continue

            explored[popped_pt] = parent

            # enqueue neighbors unexplored and unqueue with relevant priority
            neighbors = get_neighbors(popped_pt[0], popped_pt[1], model_size)

            for neighbor in neighbors:
                step_dist = euclidean_dist(self.get_position(popped_pt), self.get_position(neighbor))

                new_cost = curr_dist + step_dist

                if neighbor in enqueued:
                    queue_cost, heuristic = enqueued[neighbor]

                    if queue_cost <= new_cost:
                        continue

                else:
                    heuristic = euclidean_dist(self.get_position(mesh2), self.get_position(neighbor))

                enqueued[neighbor] = new_cost, heuristic
                heappush(queue, (new_cost + heuristic, next(c), neighbor, new_cost, popped_pt))

        return final_path, final_len



    def calculate_force(self, point1, point2):
        # Calculate the force between two points
        pass

    def update_model(self, ellipse_eccentricity: float, force_decay: float, neighbors_to_sample: int):
        # Update the force model
        self.ellipse_eccentricity = ellipse_eccentricity  # Eccentricity of the distance metric
        self.force_decay = force_decay # Rate at which the force decays with distance
        self.neighbors_to_sample = neighbors_to_sample # Neighbors to sample for the plane estimation

    def plot_model(self):
        # Plot the force model
        pass

