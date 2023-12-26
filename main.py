import torch
from algorithm import HierarchicalClustering

torch.manual_seed(2004)
points = [(-10, 8), (7, -6), (8, -10), (-6, -4), (-8, 6), (2, -4)]

points = [(9, 4), (0, 1), (9, 0), (1, 8)]
points = [(9, 0), (1, 7), (1, 7), (2, 8)]
# points = [(6, 1), (2, 3), (3, 0), (6, 1)]

hierarchical_clustering_algorithm = HierarchicalClustering()
hierarchical_clustering_algorithm.start_clustering(
                                                    points = points, 
                                                    criterion_type = "complete", # Linkage criterion type
                                                    norm_number = 2, # L1 or L2 norm
                                                    show_coords = False, # Affects print output for clusters after each iteration
                                                    )