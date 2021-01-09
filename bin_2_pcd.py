# ========================== Convert .bin file to .pcd ========================
import numpy as np
import struct
import sys
import open3d as o3d
import re

def bin2pcd(binFileName, savePCD):
    size_float = 4
    list_pcd = []
    with open(binFileName, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_pcd)
    
    if savePCD == True:
        start = binFileName.rfind('/')
        end = binFileName.rfind('.bin') 
        pcdFileName = binFileName[start+1:end] + '.pcd'
        o3d.io.write_point_cloud(pcdFileName, pcd)
    return pcd


