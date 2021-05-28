import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import pptk

from bin_2_pcd import bin2pcd


BIN_FILE_NAME = 'D:/General_Projects/LiDAR/1.bin'


# ====================== Filter out ground ====================================

def filter_ground(cloud):
    # range between min and max of z in 0.1 step
    bin_range = np.arange(cloud[:,2].min(),cloud[:,2].max(), 0.1) 
    # find distribution of points in z-axis
    counts, bins = np.histogram(cloud[:,2], bins = bin_range)
    # find for which z there is maximum points
    max_z = np.where(counts == max(counts))[0][0]
    # check that max_z + 2 won't be outside array
    if max_z + 2 <= len(bins): 
        # +2 as a factor of safety and remove some outliers, if present
        ground_z = np.round(bins[max_z + 2], 2) 
    else:
        # else just get the last bin
        ground_z = np.round(bins[-1], 2) 
    # remove ground points (anything below the ground plane)
    objects_cloud = cloud[(cloud[:,2]>ground_z)] 
    
    return objects_cloud, ground_z

# ======================== Divide cloud into zones  ===========================

def divide_zones(cloud, close_range, mid_range):
    dist = []
    # find the distances between each point and the origin/lidar
    for r in range(len(cloud)): 
        # no need for z-coord, calculate dist in xy plane
        radius = np.sqrt(sum(np.square(cloud[r][:2]))) 
        # store distances
        dist.append(radius) 
    
    dist = np.asarray(dist)
    # limit zone 1 to just the closest points
    cloud_zone1 = cloud[dist<close_range]
    # zone 2 is between the two ranges
    cloud_zone2 = cloud[(dist>close_range) & (dist<mid_range)]
    # zone 3 is the rest of the cloud
    cloud_zone3 = cloud[dist>mid_range] 
    if dist.max() < mid_range:
        print('Valid zone ranges required: within the cloud limits of [{},{}]'
              .format(dist.min(),dist.max()))
    if close_range > mid_range:
        print('Valid zone ranges required: close_range < mid_range')
        
    return cloud_zone1, cloud_zone2, cloud_zone3
    
# ===================== Cluster data with DBSCAN  =============================

def find_zone_clusters(zone_cloud, epsilon, min_points):
    zone1_pcd = o3d.geometry.PointCloud()
    zone1_pcd.points = o3d.utility.Vector3dVector(zone_cloud)
    # use DBSCAN for clustering
    cluster_labels = np.array(zone1_pcd.cluster_dbscan(eps=epsilon, 
                                                       min_points=min_points))
    
    # remove all outliers based on cluster
    cluster_labels_filtered = cluster_labels[cluster_labels!=-1]
    # also remove outlier points from cloud
    zone_cloud_filtered = zone_cloud[cluster_labels!=-1] 
    return cluster_labels_filtered, zone_cloud_filtered

# ================== Visualize clusters in color ==============================

def visualize_clusters(zone_cloud, cluster_labels):
    # create point cloud
    zone_pcd = o3d.geometry.PointCloud() 
    zone_pcd.points = o3d.utility.Vector3dVector(zone_cloud)
    
    # get the last number of the clusters
    max_label = cluster_labels.max()
    # create a different color for each cluster by splitting the 0-1 interval in n=num of clusters
    colors = plt.get_cmap("tab20")(cluster_labels/(max_label if max_label > 0 else 1)) 
    zone_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([zone_pcd])

    
if __name__ == '__main__':
    pcd = bin2pcd(BIN_FILE_NAME,False)
    # get xyz points
    cloud_xyz = np.asarray(pcd.points)
    # # visualize point cloud
    # o3d.visualization.draw_geometries([pcd]) 
    # v0 = pptk.viewer(cloud_xyz)
    
    # filter out the ground points
    objects_cloud_xyz, ground_z = filter_ground(cloud_xyz) 
    
    # split cloud into zones
    object_cloud_z1, object_cloud_z2, object_cloud_z3 = divide_zones(objects_cloud_xyz, 15, 30) 
    
    # create zones and update clouds based on clustering
    
    cluster_z1_eps = 0.5
    cluster_z2_eps = 0.8
    cluster_z3_eps = 3
    cluster_z1_min_points = 50
    cluster_z2_min_points = 40
    cluster_z3_min_points = 30
    
    cluster_labels_z1, object_cloud_z1_f = find_zone_clusters(object_cloud_z1, cluster_z1_eps, cluster_z1_min_points)
    # visualize_clusters(object_cloud_z1_f, cluster_labels_z1)
    
    cluster_labels_z2, object_cloud_z2_f = find_zone_clusters(object_cloud_z2, cluster_z2_eps, cluster_z2_min_points)
    # visualize_clusters(object_cloud_z2_f, cluster_labels_z2)
    
    cluster_labels_z3, object_cloud_z3_f = find_zone_clusters(object_cloud_z3, cluster_z3_eps, cluster_z3_min_points)
    # visualize_clusters(object_cloud_z3_f, cluster_labels_z3)
    
    
    # shift all labels by num of z1 labels
    cluster_labels_z2 = cluster_labels_z2 + (cluster_labels_z1.max() + 1)
    # shift all labels by num of new z2 labels
    cluster_labels_z3 = cluster_labels_z3 + (cluster_labels_z2.max() + 1)
    # concatenate all the labels
    all_cluster_labels = np.concatenate((cluster_labels_z1, cluster_labels_z2))# , cluster_labels_z3))
    # combine all point clouds for the 3 zones
    clustered_cloud_xyz = np.concatenate((object_cloud_z1_f, object_cloud_z2_f))#, object_cloud_z3_f)) 
    
    visualize_clusters(clustered_cloud_xyz, all_cluster_labels)

