import torch
from functions import calc_l1_norm, calc_l2_norm
from cluster import Cluster
from algorithm import HierarchicalClustering

torch.manual_seed(2004)
K = 3 # Number of clusters
points = [(9, 4), (0, 1), (9, 0), (1, 8)]
points = [(-10, 8), (7, -6), (8, -10), (-6, -4), (-8, 6), (2, -4)]

num_points = len(points)
for i in range(0, num_points):
    points[i] = torch.tensor(points[i])

# Find L1 and L2 norms for each point with every other point
euclid_dist_hashmap = {}
manhattan_dist_hashmap = {}

for i in range(0, num_points):
    current_point = points[i]
    for j in range(i + 1, num_points):
        other_point = points[j]

        distance_between_points_l1 = calc_l1_norm(p1 = other_point, p2 = current_point) # Manhattan distance
        distance_between_points_l2 = calc_l2_norm(p1 = other_point, p2 = current_point) # Euclidean distance

        manhattan_dist_hashmap[(i, j)] =  distance_between_points_l1
        manhattan_dist_hashmap[(j, i)] = distance_between_points_l1

        euclid_dist_hashmap[(i, j)] =  distance_between_points_l2
        euclid_dist_hashmap[(j, i)] = distance_between_points_l2

# Create clusters, one for each point
clusters = []
for i, point in enumerate(points):
    new_cluster = Cluster(cluster_number = i)
    new_cluster.add_to_cluster(point = point)
    clusters.append(new_cluster)
    print(i, new_cluster.points)

hierarchical_clustering_algorithm = HierarchicalClustering()
hierarchical_clustering_algorithm.start_clustering(
                                                    clusters = clusters,
                                                    points = points, 
                                                    l1_hashmap = manhattan_dist_hashmap, 
                                                    l2_hashmap = euclid_dist_hashmap, 
                                                    criterion_type = "complete"
                                                    )