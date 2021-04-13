import argparse
import os

import numpy as np
import trimesh


def show(y, mesh):
    colors = np.zeros([y.shape[0], 4])
    colors[y == 0] = [0, 255, 0, 255]
    colors[y == 1] = [255, 0, 0, 255]

    mesh = trimesh.base.Trimesh(mesh.vertices, mesh.faces, face_colors=colors)
    mesh.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HolePrediction')
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()

    y = np.loadtxt(os.path.join(args.path, 'y.xyz'))
    y_ = np.loadtxt(os.path.join(args.path, 'y_.xyz'))
    mesh = trimesh.load(os.path.join(args.path, 'x.stl'))

    show(y, mesh)
    show(y_, mesh)
