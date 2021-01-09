import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pptk
import random
import time
from bin_2_pcd import bin2pcd

# ====================== Filter out ground ====================================

def filter_ground(cloud):
    bin_range = np.arange(cloud[:,2].min(),cloud[:,2].max(), 0.1) # range between min and max of z in 0.1 step
    counts, bins = np.histogram(cloud[:,2], bins = bin_range) # find distribution of points in z-axis
    max_z = np.where(counts == max(counts))[0][0] # find for which z there is maximum points
    if max_z + 2 <= len(bins): # check that max_z + 2 won't be outside array
        ground_z = np.round(bins[max_z + 2], 2) # +2 is in order as a factor of safety and remove some outliers, if present
    else:
        ground_z = np.round(bins[-1], 2) # else just get the last bin
    objects_cloud = cloud[(cloud[:,2]>ground_z)] # remove ground points (anything below the ground plane)
    
    return objects_cloud, ground_z

# ======================== Divide cloud into zones  ===========================

def divide_zones(cloud, close_range, mid_range):
    dist = []
    for r in range(len(cloud)): # find the distances between each point and the origin/lidar
        radius = np.sqrt(sum(np.square(cloud[r][:2]))) # no need for z-coord, calculate dist in xy plane
        dist.append(radius) # store distances
    
    dist = np.asarray(dist)
    cloud_zone1 = cloud[dist<close_range] # limit zone 1 to just the closest points
    cloud_zone2 = cloud[(dist>close_range) & (dist<mid_range)] # zone 2 is between the two ranges
    cloud_zone3 = cloud[dist>mid_range] # zone 3 is the rest of the cloud
    if dist.max() < mid_range:
        print('Valid zone ranges required: within the cloud limits of [{},{}]'.format(dist.min(),dist.max()))
    if close_range > mid_range:
        print('Valid zone ranges required: close_range < mid_range')
        
    return cloud_zone1, cloud_zone2, cloud_zone3
    
# ===================== Cluster data with DBSCAN  =============================

def find_zone_clusters(zone_cloud, epsilon, min_points):
    zone1_pcd = o3d.geometry.PointCloud()
    zone1_pcd.points = o3d.utility.Vector3dVector(zone_cloud)
    cluster_labels = np.array(zone1_pcd.cluster_dbscan(eps=epsilon, min_points=min_points))# use DBSCAN for clustering
    
    cluster_labels_filtered = cluster_labels[cluster_labels!=-1] # remove all outliers based on cluster
    zone_cloud_filtered = zone_cloud[cluster_labels!=-1] # also remove outlier points from cloud
    return cluster_labels_filtered, zone_cloud_filtered

# ================== Visualize clusters in color ==============================

def visualize_clusters(zone_cloud, cluster_labels):
    zone_pcd = o3d.geometry.PointCloud() # create point cloud
    zone_pcd.points = o3d.utility.Vector3dVector(zone_cloud)
    
    max_label = cluster_labels.max() # get the last number of the clusters
    colors = plt.get_cmap("tab20")(cluster_labels/(max_label if max_label > 0 else 1)) # create a different color for each cluster by splitting the 0-1 interval in n=num of clusters
    zone_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([zone_pcd])
#%%
    
# ================== Create point cloud from data =============================
binFileName = '.../1.bin' # choose bin path
pcd = bin2pcd(binFileName,False) 
# pcd = o3d.io.read_point_cloud(pcd_file) # create pcd object from file
cloud_xyz = np.asarray(pcd.points) # get xyz points
o3d.visualization.draw_geometries([pcd]) # visualize point cloud
v0 = pptk.viewer(cloud_xyz)

# ===============================    Main    ==================================
objects_cloud_xyz, ground_z = filter_ground(cloud_xyz) # filter out the ground points

object_cloud_z1, object_cloud_z2, object_cloud_z3 = divide_zones(objects_cloud_xyz, 15, 30) # split cloud into zones
#%%
# create zones and update clouds based on clustering

cluster_z1_eps = 0.5
cluster_z2_eps = 0.8
cluster_z3_eps = 3
cluster_z1_min_points = 50
cluster_z2_min_points = 40
cluster_z3_min_points = 30

cluster_labels_z1, object_cloud_z1_f = find_zone_clusters(object_cloud_z1, cluster_z1_eps, cluster_z1_min_points)
visualize_clusters(object_cloud_z1_f, cluster_labels_z1)

cluster_labels_z2, object_cloud_z2_f = find_zone_clusters(object_cloud_z2, cluster_z2_eps, cluster_z2_min_points)
visualize_clusters(object_cloud_z2_f, cluster_labels_z2)

cluster_labels_z3, object_cloud_z3_f = find_zone_clusters(object_cloud_z3, cluster_z3_eps, cluster_z3_min_points)
visualize_clusters(object_cloud_z3_f, cluster_labels_z3)

# =============================================================================

cluster_labels_z2 = cluster_labels_z2 + (cluster_labels_z1.max() + 1) # shift all labels by num of z1 labels
cluster_labels_z3 = cluster_labels_z3 + (cluster_labels_z2.max() + 1)  # shift all labels by num of new z2 labels
all_cluster_labels = np.concatenate((cluster_labels_z1, cluster_labels_z2))# , cluster_labels_z3)) # concatenate all the labels
clustered_cloud_xyz = np.concatenate((object_cloud_z1_f, object_cloud_z2_f))#, object_cloud_z3_f)) # combine all point clouds for the 3 zones

visualize_clusters(clustered_cloud_xyz, all_cluster_labels)

