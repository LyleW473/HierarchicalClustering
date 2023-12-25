import torch
from cluster import Cluster

class HierarchicalClustering:
    
    def create_dist_hashmap(self, points, norm_number):

        dist_hashmap = {}
        num_points = len(points)
        norm_calc_function = self.select_norm_calc_function(norm_number)

        for i in range(0, num_points):
            current_point = points[i]
            for j in range(i + 1, num_points):
                other_point = points[j]
                # Find distance between points using the specified norm (L1 or L2 norm)
                distance_between_points = norm_calc_function(p1 = other_point, p2 = current_point)
                dist_hashmap[(i, j)] =  distance_between_points
                dist_hashmap[(j, i)] = distance_between_points
        
        return dist_hashmap
            
    def select_norm_calc_function(self, norm_number):
        return getattr(self, f"calc_l{norm_number}_norm")

    # L1 Norm - Manhattan distance 
    def calc_l1_norm(self, p1, p2):
        return abs(p2[0] - p1[0]) + abs(p2[1] - p1[1])

    # L2 Norm - Euclidean distance
    def calc_l2_norm(self, p1, p2):
        return (((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2))**0.5

    def create_initial_clusters(self, points):

        # Create clusters, one for each point
        clusters = []
        for i, point in enumerate(points):
            new_cluster = Cluster(cluster_number = i)
            new_cluster.add_to_cluster(point = point)
            clusters.append(new_cluster)
            print(i, new_cluster.points)
        print()

        return clusters
    
    def select_linkage_criterion(self, criterion_type):
        return getattr(self, f"{criterion_type}_linkage_criterion")

    def single_linkage_criterion(self, distances):
        return min(distances)

    def complete_linkage_criterion(self, distances):
        return max(distances)
    
    def average_linkage_criterion(self, distances):
        return (1/len(distances)) * sum(distances)
    
    def merge_clusters(self, clusters, clusters_to_merge):
        
        # Find all the clusters that need to be merged
        cluster_indexes = set()
        for cluster_pair in clusters_to_merge:
            cluster_indexes.add(cluster_pair[0])
            cluster_indexes.add(cluster_pair[1])

        # Select the cluster with the minimum cluster index as the one to move all points to
        min_cluster_index = min(cluster_indexes)
        cluster_to_add_points_to = clusters[min_cluster_index]
        cluster_indexes.remove(min_cluster_index) # Remove min cluster index (as it is the merged cluster)
        
        # Add all points from the other clusters to the cluster with the minimum cluster index
        for cluster_index in cluster_indexes:
            for point_to_add in clusters[cluster_index].points:
                cluster_to_add_points_to.add_to_cluster(point_to_add)

        # Return new list of clusters
        new_clusters = []
        for k in range(0, len(clusters)):
            if k not in cluster_indexes:
                new_clusters.append(clusters[k])
        return new_clusters
    
    def initialise_points(self, points):
        # Convert points from tuples to tensors
        num_points = len(points)
        for i in range(0, num_points):
            points[i] = torch.tensor(points[i])

    def start_clustering(self, points, criterion_type, norm_number):
        
        self.initialise_points(points)
        dist_hashmap = self.create_dist_hashmap(points, norm_number)
        linkage_criterion = self.select_linkage_criterion(criterion_type)
        clusters = self.create_initial_clusters(points)

        while len(clusters) > 1:
            
            # Maps (cluster1, cluster2) to the distance between the two clusters, defined by the linkage criterion used
            clusters_distances = {}
            min_distance = float("inf")

            for i in range(0, len(clusters)):
                for j in range(i + 1, len(clusters)):

                    distances_between_cluster_pair = []
                    # For each point in each of the cluster, find the distance between them
                    for p_i in clusters[i].points:
                        for p_j in clusters[j].points:
                            
                            # Find index of points to reference hashmap
                            for x in range(len(points)):
                                if torch.equal(points[x], p_i):
                                    p_i_index = x
                                elif torch.equal(points[x], p_j):
                                    p_j_index = x
                            
                            # Find distance between points
                            distance_between_points = dist_hashmap[(p_i_index, p_j_index)]
                            distances_between_cluster_pair.append(distance_between_points)
                    
                    # Select the distance based on the linkage criterion            
                    selected_distance = linkage_criterion(distances_between_cluster_pair)

                    # Save the distance between these 2 clusters
                    clusters_distances[(i, j)] = selected_distance
                    min_distance = min(min_distance, selected_distance) # Update the minimum distance

            # Merge all clusters with the minimum distance
            clusters_to_merge = [cluster_pair for cluster_pair in clusters_distances if clusters_distances[cluster_pair] == min_distance]
            clusters = self.merge_clusters(clusters = clusters, clusters_to_merge = clusters_to_merge)

            # Display information
            for k, cluster in enumerate(clusters):
                print(f"Cluster {k}: {[(point[0].item(), point[1].item()) for point in cluster.points]}")
            print()