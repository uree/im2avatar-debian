import numpy as np
import open3d as o3d
import h5py
import sys
import os


def save_obj(pcd, filepath, file_name, format=".ply"):
    outpath = filepath+file_name.split("/")[-1].split(".")[0]+format
    o3d.io.write_point_cloud(outpath, pcd)
    print("<Done> Object saved as ", outpath)
    return outpath


def obj_display(file_name, output_folder, save=True, display=False):
    threshold = 0.8

    model = h5py.File(file_name, 'r')

    data = np.array(model['data'][:])

    X = data[:,:,:,0]

    shape_pred = np.where(data > threshold, np.ones_like(data), np.zeros_like(data))
    shape_pred = shape_pred.astype(np.int)

    occurrences = np.count_nonzero(shape_pred == 0)
    coordinates = np.argwhere(shape_pred != 0)

    # this fails sometimes. the shape of the data can be unpredictable
    ordernates = coordinates[:,1:-1]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ordernates)

    if display:
        o3d.visualization.draw_geometries([pcd])

    if save:
        save_obj(pcd, output_folder, file_name)
