# lidar-clustering
The Python file takes in a .bin file with LiDAR data and performs a DBSCAN clustering in order to find the defferent objects present.  The clustering is visualized with all clusters/objects having different colors. The script is taylored to data coming from vehicles, as the ground filtering is based on the premise that the ground is rougly bound to one plane. Most of the code is automated, yet for the partiular data, the DBSCAN algorithm parameters need to be tuned.
