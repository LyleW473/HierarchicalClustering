import torch

class Cluster:

    def __init__(self, cluster_number):
        self.cluster_number = cluster_number
        self.points = []
    
    # Adds point to cluster
    def add_to_cluster(self, point_index):
        self.points.append(point_index)