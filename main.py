import torch

torch.manual_seed(2004)
K = 3 # Number of clusters
points = [(9, 4), (0, 1), (9, 0), (1, 8)]
points = set([torch.tensor(point) for point in points])


# Find points between each other point
