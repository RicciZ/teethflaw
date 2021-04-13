from __future__ import print_function

import argparse
import os, math
import shutil

import numpy as np
import pandas as pd
import pickle
import json
import torch
import torch.nn as nn
from data import MeshHoleDataset
from model import get_model

def dp_in_test(m):
    if isinstance(m, nn.Dropout):
        m.train()

def test(args):
    dataset = MeshHoleDataset(args.pc_root, args.mesh_root, args.tooth_ids)
    # Try to load models
    print(args.model_path)
    device = torch.device("cuda")
    model = nn.DataParallel(get_model(args.model, args))
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    if not args.save_path:
        save_path = os.path.join(os.path.dirname(os.path.dirname(args.model_path)), 'results')
    else:
        save_path = args.save_path
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    # forward model
    model.eval()

    # enable dropout in test
    model.apply(dp_in_test)
    accs, pos_accs, neg_accs = [], [], []
    fp_accs, fn_accs = [], []
    mIoU = []
    col_names = ['PatientID', 'Acc', 'Pos_Acc', 'Neg_Acc', "FP_Acc", "FN_Acc", "mIoU"]
    my_df = pd.DataFrame(columns=col_names)
    with torch.no_grad():
        for x, y, mesh, path in dataset:
            index = np.arange(x.shape[0])
            np.random.shuffle(index)
            x = x[index, :]
            y = y[index]
            size_x = x.shape[0]

            # create a dict to shuffle back
            index_dict = {}
            for i, j in enumerate(index):
                index_dict[j] = i

            n_iter = math.ceil(size_x / args.n_point)
            y_ = []
            prob = []
            for i in range(n_iter):
                if i < n_iter - 1:
                    x_in = x[i*args.n_point:((i+1)*args.n_point), :]
                    x_in = torch.tensor(x_in).to(device).unsqueeze(0).permute(0, 2, 1)
                    f = model(x_in)
                    y_out = f.argmax(dim=1).cpu().numpy()[0]
                    prob_out = f.cpu().numpy()[0] # 2 * size_x
                    y_.append(y_out)
                    prob.append(prob_out)
                else:
                    x_in = np.concatenate((x[i*args.n_point:,:], x[:((i+1)*args.n_point-size_x)]), axis=0)
                    x_in = torch.tensor(x_in).to(device).unsqueeze(0).permute(0, 2, 1)
                    f = model(x_in)
                    y_out = f.argmax(dim=1).cpu().numpy()[0]
                    prob_out = f.cpu().numpy()[0]
                    y_.append(y_out)
                    prob.append(prob_out)

            y_ = np.asarray(y_).reshape(-1)[:size_x]
            prob = np.hstack(prob)[:, :size_x]
            accs.append(np.mean(y == y_))
            pos_accs.append(np.mean(y_[y == 1] == 1))
            neg_accs.append(np.mean(y_[y == 0] == 0))
            fp_accs.append(np.mean(y_[y == 0] == 1))
            fn_accs.append(np.mean(y_[y == 1] == 0))
            mIoU.append(np.mean(np.logical_and(y, y_))/np.mean(np.logical_or(y, y_)))
            my_df.loc[len(my_df)] = [path, accs[-1], pos_accs[-1], neg_accs[-1], fp_accs[-1], fn_accs[-1], mIoU[-1]]
            save_dir = os.path.join(save_path, path)
            if not os.path.isdir(save_dir):
                tid = path.split('_')[-1]
                shutil.copytree(os.path.join(args.mesh_root, str(tid), path), save_dir, symlinks=False, ignore=None)

            yy_ = np.zeros(size_x,dtype=np.long)
            yy = np.zeros(size_x,dtype=np.long)
            pprob = np.zeros((2, size_x),dtype=np.float)
            for i in range(size_x):
                yy_[i] = y_[index_dict[i]]
                yy[i] = y[index_dict[i]]
                pprob[:, i] = prob[:, index_dict[i]]

            np.savetxt(os.path.join(save_dir, 'y_.xyz'), yy_)
            np.savetxt(os.path.join(save_dir, 'prob.xyz'), pprob)
            np.savetxt(os.path.join(save_dir, 'y.xyz'), yy)
            mesh.export(os.path.join(save_dir, 'x.stl'))

            print('Path {}, mIoU: {}, Acc: {:.4f}, POS_ACC: {:.4f}, NEG_ACC: {:.4f}, FP_ACC: {:.4f}, FN_ACC: {:.4f}'.format(
                path, mIoU[-1], accs[-1], pos_accs[-1], neg_accs[-1], fp_accs[-1], fn_accs[-1]))
    my_df.to_pickle(os.path.join(save_path, 'accs.pkl'))
    # print('Acc:', accs)
    # print('Pos_acc:', pos_accs)
    # print('neg_acc:', neg_accs)
    print(np.mean(mIoU), np.mean(accs), np.mean(pos_accs), np.mean(neg_accs), np.mean(fp_accs), np.mean(fn_accs))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HolePrediction')
    parser.add_argument('--model', type=str, default='dgcnn', choices=['dgcnn'],
                        help='Model to use, [dgcnn]')
    parser.add_argument('--k', type=int, default=20,
                        help='Num of nearest neighbors to use')
    parser.add_argument('--p', type=float, default=0.5,
                        help='dropout ratio for bayesian estimate')
    parser.add_argument('--n_point', type=int, default=8000,
                        help='Num of points to use')
    parser.add_argument('--all', type=bool, default=False,
                        help='Use all teeth or not')
    parser.add_argument('--save_path', type=str, default=None,
                        help='save results path')
    parser.add_argument('--tooth_ids', type=str, default='36', help='tooth ids')
    parser.add_argument('--model_path', type=str, required=True, help='model path')
    parser.add_argument('--pc_root', type=str, default='std_split_data/valid', required=True, help='point cloud root')
    parser.add_argument('--mesh_root', type=str,default='data/bad/', required=True, help='mesh root')

    args = parser.parse_args()

    if args.all:
        args.tooth_ids = [13,14,15,16,17,23,24,25,26,27,28,33,34,35,36,37,43,44,45,46,47]
    else:
        args.tooth_ids = [int(i) for i in args.tooth_ids.split(',')]
    print(args)

    test(args)
