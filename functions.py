# L1 Norm - Manhattan distance 
def calc_l2_norm(p1, p2):
    return abs(p2[0] - p1[0]) + abs(p2[1] - p1[1])

# L2 Norm - Euclidean distance
def calc_l2_norm(p1, p2):
    return (((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2))**0.5