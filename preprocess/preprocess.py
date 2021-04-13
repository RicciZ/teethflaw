import argparse
import glob
import os
import re
from functools import partial
from multiprocessing import Pool

import numpy as np
import tqdm
import trimesh
from scipy.spatial.transform import Rotation as R


def extract_transform_info(id, path):
    with open(path) as f:
        for line in f:
            if line.startswith(id):
                params = [float(p) for p in line.split()[1:]]
                return params


def to_matrix(params):
    M = np.zeros(shape=(4, 4))
    M[:3, 3] = params[4:]
    M[:3, :3] = R.from_quat(params[0:4]).as_matrix()
    M[3, 3] = 1

    return np.linalg.inv(M)

def _get_corner(mesh):
    # each face has three vertices v1, v2, v3, and a center c
    # get (v1-c), (v2-c), (v3-c)
    return (mesh.triangles - np.expand_dims(mesh.triangles_center, axis=1)).reshape(-1, 9)


def _preprocess(folder, root, save_dir, tol=-0.05):
    x_path = glob.glob(os.path.join(root, folder, 'tooth*.stl'))[0]
    y_path = glob.glob(os.path.join(root, folder, '*teeth_hole*.stl'))

    # mesh load & union
    x_mesh = trimesh.load(x_path)
    y_mesh = trimesh.load(y_path[0])
    if len(y_path) > 1:
        for mesh_i in y_path[1:]:
            y_mesh += trimesh.load(mesh_i)

    # world --> local
    trans_maxtrix = to_matrix(
        extract_transform_info(re.compile('\d+').findall(x_path)[-1], os.path.join(root, folder, 'TeethAxis.txt')))
    x_mesh = x_mesh.apply_transform(trans_maxtrix)
    y_mesh = y_mesh.apply_transform(trans_maxtrix)

    # unit norm
    avg_v = x_mesh.vertices.mean(0)
    std_v = x_mesh.vertices.std(0)
    x_mesh.vertices = (x_mesh.vertices - avg_v) / std_v
    y_mesh.vertices = (y_mesh.vertices - avg_v) / std_v

    # label
    vertices = x_mesh.triangles_center
    x_mesh.fix_normals()
    normals = x_mesh.face_normals
    dist = np.asarray(trimesh.proximity.signed_distance(y_mesh, vertices))
    is_hole = dist > tol
    corner = _get_corner(x_mesh)

    #x = np.hstack([vertices, normals]).astype(np.float32)
    x = np.hstack((vertices, normals, corner)).astype(np.float32)
    y = is_hole.astype(np.long)

    # mkdir
    save_folder = os.path.join(save_dir, folder)
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    # export
    np.savetxt(os.path.join(save_folder, 'x.xyz'), x)
    np.savetxt(os.path.join(save_folder, 'y.xyz'), y)


def preprocess(root, save_dir, n_thread=4):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    #folders = [f for f in os.listdir(root) if f.startswith('DC')]
    folders = [f for f in os.listdir(root) if f.startswith('C')]
    with Pool(n_thread) as p:
        list(tqdm.tqdm(p.imap_unordered(partial(_preprocess, root=root, save_dir=save_dir), folders),total=len(folders)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HolePrediction')
    parser.add_argument('--data', type=str, default='../data/sample',
                        help='path to raw stl data')
    parser.add_argument('--save', type=str, default='../data/std_data',
                        help='path to save')
    parser.add_argument('--id', type=int, default=11,
                        help='tooth id')
    parser.add_argument('--all', type=bool, default=False,
                        help='Use all teeth or not')
    args = parser.parse_args()

    if not os.path.isdir(args.save):
        os.makedirs(args.save)
    if args.all:
        args.id = [11,12,13,14,15,16,17,21,22,23,24,25,26,27,28,31,32,33,34,35,36,37,38,41,42,43,44,45,46,47,48]
        for id in args.id:
            preprocess(os.path.join(args.data, str(id)), os.path.join(args.save, str(id)))
    else:
        preprocess(os.path.join(args.data, str(args.id)), os.path.join(args.save, str(args.id)))
