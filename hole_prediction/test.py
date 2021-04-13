from __future__ import print_function

import argparse
import os

import numpy as np
import pandas as pd
import pickle
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
    col_names = ['PatientID', 'Acc', 'Pos_Acc', 'Neg_Acc', "FP_Acc", "FN_Acc"]
    my_df = pd.DataFrame(columns=col_names)
    with torch.no_grad():
        for x, y, mesh, path in dataset:
            x = torch.tensor(x).to(device).unsqueeze(0).permute(0, 2, 1)
            f = model(x)
            prob = f.cpu().numpy()[0]
            y_ = f.argmax(dim=1).cpu().numpy()[0]
            accs.append(np.mean(y == y_))
            pos_accs.append(np.mean(y_[y == 1] == 1))
            neg_accs.append(np.mean(y_[y == 0] == 0))
            fp_accs.append(np.mean(y_[y == 0] == 1))
            fn_accs.append(np.mean(y_[y == 1] == 0))
            my_df.loc[len(my_df)] = [path, accs[-1], pos_accs[-1], neg_accs[-1], fp_accs[-1], fn_accs[-1]]
            save_dir = os.path.join(save_path, path)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            np.savetxt(os.path.join(save_dir, 'y_.xyz'), y_)
            np.savetxt(os.path.join(save_dir, 'prob_.xyz'), prob)
            np.savetxt(os.path.join(save_dir, 'y.xyz'), y)
            mesh.export(os.path.join(save_dir, 'x.stl'))
            print('Path {}, Acc: {}, POS_ACC: {}, NEG_ACC: {}, FP_ACC: {}, FN_ACC: {}'.format(
                path, accs[-1], pos_accs[-1], neg_accs[-1], fp_accs[-1], fn_accs[-1]))
    my_df.to_pickle(os.path.join(save_path, 'accs.pkl'))
    print('Acc:', accs)
    print('Pos_acc:', pos_accs)
    print('neg_acc:', neg_accs)
    print(np.mean(accs), np.mean(pos_accs), np.mean(neg_accs), np.mean(fp_accs), np.mean(fn_accs))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HolePrediction')
    parser.add_argument('--model', type=str, default='dgcnn', choices=['dgcnn'],
                        help='Model to use, [dgcnn]')
    parser.add_argument('--k', type=int, default=20,
                        help='Num of nearest neighbors to use')
    parser.add_argument('--p', type=float, default=0.5,
                        help='dropout ratio for bayesian estimate')
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
        #args.tooth_ids = [11,12,13,14,15,16,17,21,22,23,24,25,26,27,31,32,33,34,35,36,37,41,42,43,44,45,46,47]
        args.tooth_ids = [13,14,15,16,17,23,24,25,26,27,33,34,35,36,37,43,44,45,46,47]
    else:
        args.tooth_ids = [int(i) for i in args.tooth_ids.split(',')]
    print(args)

    test(args)
