import argparse
import json
import os

import numpy as np
import tqdm


def calc_stat(root, tooth_ids, save_dir):
    labels = [os.path.join(root, str(id), f, 'y.xyz') for id in tooth_ids for f in
              os.listdir(os.path.join(root, str(id)))]
    print(labels)
    pos_num, neg_num = 0, 0
    for l in tqdm.tqdm(labels):
        y = np.loadtxt(l)
        pos_num += np.sum(y == 1)
        neg_num += np.sum(y == 0)

    pos_rate = pos_num / float(pos_num + neg_num)

    if len(tooth_ids) >= 10:
        path = os.path.join(save_dir, 'stat_all.json')
    else:
        path = os.path.join(save_dir, 'stat_{}.json'.format('_'.join(map(str, tooth_ids))))
    with open(path, 'w') as json_file:
        json.dump({'pos_rate': pos_rate,
                   'neg_rate': 1 - pos_rate}, json_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calc stats')
    parser.add_argument('--data', type=str, default='../data/std_split_data/train',
                        help='path to xyz train data')
    parser.add_argument('--id', nargs='+', type=int, default=[11],
                        help='tooth ids')
    parser.add_argument('--all', type=bool, default=False,
                        help='Use all teeth or not')
    args = parser.parse_args()
    if args.all:
        args.id = [11,12,13,14,15,16,17,21,22,23,24,25,26,27,28,31,32,33,34,35,36,37,38,41,42,43,44,45,46,47,48]
    calc_stat(args.data, args.id, '../data')
