# lidar-clustering
The lidar_clustering Python script takes in a .bin file with LiDAR data and performs a DBSCAN clustering in order to find the defferent objects present.  The clustering is visualized with all clusters/objects having different colors. The script is taylored to data coming from vehicles, such as autonomous cars, as the ground filtering is based on the premise that the ground is rougly bound to one plane. Most of the code is automated, yet for the partiular data, the DBSCAN algorithm parameters need to be tuned.

All scripts have comments, but the Jupyter notebook has more in-depth explanations for each step, as well as tips on how to adjust the DBSCAN parameters for a given scenario. 

The 'lidar_clustering_example.png' shows the result of clustering a random image from the KITTI Dataset. The different parked cars can be seen as distinct objects in different colors, and the buildings and other objects around the road are also clustered. 
