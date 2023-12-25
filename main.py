import torch
from functions import calc_l1_norm, calc_l2_norm
from cluster import Cluster

torch.manual_seed(2004)
K = 3 # Number of clusters
points = [(9, 4), (0, 1), (9, 0), (1, 8)]
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
        
print(euclid_dist_hashmap)
print(manhattan_dist_hashmap)

# Create clusters, one for each point
clusters = []
for i, point in enumerate(points):
    new_cluster = Cluster(cluster_number = i)
    new_cluster.add_to_cluster(point = point)
    clusters.append(new_cluster)
    print(i, new_cluster.points)