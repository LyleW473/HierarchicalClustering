import torch

class HierarchicalClustering:

    def start_clustering(self, clusters, points, criterion_type, l1_hashmap, l2_hashmap):

        linkage_criterion = self.select_linkage_criterion(criterion_type)

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
                            distance_between_points = l1_hashmap[(p_i_index, p_j_index)]
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

    def select_linkage_criterion(self, criterion_type):
        return getattr(self, f"{criterion_type}_linkage_criterion")

    def single_linkage_criterion(self, distances):
        return min(distances)

    def complete_linkage_criterion(self, distances):
        return max(distances)
    
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