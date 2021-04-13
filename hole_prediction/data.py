import os

import numpy as np
import trimesh
from torch.utils.data import Dataset


def pc_jitter(x, sigma=0.01, clip=0.02):
    x += np.clip(sigma * np.random.randn(*x.shape), -clip, clip)

    return x


def pc_augment(x):
    return pc_jitter(x)


class HoleDataset(Dataset):
    def __init__(self, root, tooth_ids, n_point=8000, train=True):
        self.train = train
        self.n_point = n_point
        self.data = [(os.path.join(root, str(id), f, 'x.xyz'),
                      os.path.join(root, str(id), f, 'y.xyz')) for id in tooth_ids for f in
                     os.listdir(os.path.join(root, str(id)))]

    def __getitem__(self, item):
        x = np.loadtxt(self.data[item][0])
        y = np.loadtxt(self.data[item][1])
        path = self.data[item][0].split('/')[-2]

        id = np.random.choice(x.shape[0], self.n_point)
        x, y = x[id, :], y[id]

        if self.train:
            x = pc_augment(x)

        return x.astype(np.float32), y.astype(np.long), path

    def __len__(self):
        return len(self.data)

class HoleDataset_bad(Dataset):
    def __init__(self, root, tooth_ids, n_point=8000, train=True):
        self.train = train
        self.n_point = n_point
        self.data = [(os.path.join(root, str(id), f, 'x.xyz'),
                      os.path.join(root, str(id), f, 'y.xyz')) for id in tooth_ids for f in
                     os.listdir(os.path.join(root, str(id))) if 'good' not in f]

    def __getitem__(self, item):
        x = np.loadtxt(self.data[item][0])
        y = np.loadtxt(self.data[item][1])
        path = self.data[item][0].split('/')[-2]

        id = np.random.choice(x.shape[0], self.n_point)
        x, y = x[id, :], y[id]

        if self.train:
            x = pc_augment(x)

        return x.astype(np.float32), y.astype(np.long), path

    def __len__(self):
        return len(self.data)


class MeshHoleDataset(Dataset):
    def __init__(self, pc_root, mesh_root, tooth_ids):
        self.data = [(os.path.join(pc_root, str(id), f, 'x.xyz'),
                      os.path.join(pc_root, str(id), f, 'y.xyz'),
                      os.path.join(mesh_root, str(id), f, 'tooth{}.stl'.format(id))) for id in tooth_ids for f in
                     os.listdir(os.path.join(pc_root, str(id)))]

    def __getitem__(self, item):
        x = np.loadtxt(self.data[item][0])[:, :]
        y = np.loadtxt(self.data[item][1])
        mesh = trimesh.load(self.data[item][2])
        path = self.data[item][0].split('/')[-2]

        return x.astype(np.float32), y.astype(np.long), mesh, path

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    train = HoleDataset('std_split_data/train', [36, 36])

    for data, label, path in train:
        print(path)
        print(data.shape)
        print(label.shape)
        print(data.max(), data.min(), data.mean(), data.std())
        print(label.max(), label.min(), label.mean(), label.std())
    # test = MeshHoleDataset('std_split_data/valid', 'data', [36])
    #
    # for x, y, mesh, path in test:
    #     colors = np.zeros([x.shape[0], 4])
    #     colors[y == 0] = [0, 255, 0, 255]
    #     colors[y == 1] = [255, 0, 0, 255]
    #
    #     mesh = trimesh.base.Trimesh(mesh.vertices, mesh.faces, face_colors=colors)
    #     mesh.show()
