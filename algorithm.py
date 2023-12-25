import torch

class HierarchicalClustering:
    def __init__(self):
        pass

    def start_clustering(self, clusters, points, criterion_type, l1_hashmap, l2_hashmap):


        complete = False
        n_clusters = len(clusters)
        min_distance = float("inf")
        linkage_criterion = self.select_linkage_criterion(criterion_type)

        while not complete:
            
            # Maps (cluster1, cluster2) to the distance between the two clusters, defined by the linkage criterion used
            clusters_distances = {}

            for i in range(0, n_clusters):
                print("C_DIST", clusters_distances)
                for j in range(i + 1, n_clusters):

                    distances_between_cluster_pair = {}
                    # For each point in each of the cluster, find the distance between them
                    for p_i in clusters[i].points:
                        for p_j in clusters[j].points:
                            
                            # Find index of points to reference hashmap
                            for x in range(len(points)):
                                if torch.equal(points[x], p_i):
                                    p_i_index = x
                                elif torch.equal(points[x], p_j):
                                    p_j_index = x

                            print(p_i_index, p_j_index, l1_hashmap[(p_i_index, p_j_index)])

                            # Find distance between points
                            distance_between_points = l1_hashmap[(p_i_index, p_j_index)]
                            distances_between_cluster_pair[j] = distance_between_points
                    
                    # Select the distance based on the linkage criterion            
                    selected_index = linkage_criterion(distances_between_cluster_pair)
                    print(selected_index, distances_between_cluster_pair[selected_index])
                    selected_distance = distances_between_cluster_pair[selected_index]

                    # Save the distance between these 2 clusters
                    print("A", distances_between_cluster_pair, selected_distance, (i, j))
                    clusters_distances[(i, j)] = selected_distance
                    min_distance = min(min_distance, selected_distance) # Update the minimum distance

            # Merge all clusters with the minimum distance
            clusters_to_merge = [cluster_pair for cluster_pair in clusters_distances if clusters_distances[cluster_pair] == min_distance]
            print(clusters_distances)
            print(clusters_to_merge)
            print(min_distance)
            print()


    def select_linkage_criterion(self, criterion_type):
        return getattr(self, f"{criterion_type}_linkage_criterion")

    def single_linkage_criterion(self, distances):
        return min(distances, key = distances.get)

    def complete_linkage_criterion(self, distances):
        return max(distances, key = distances.get)
