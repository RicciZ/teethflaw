import argparse
import itertools
import os
import random
import shutil


def partition_data(root, dst, tooth_id, val_ratio=0.1, test_ratio=0.2):
    samples = get_tooth_set(root)
    keys = list(samples.keys())
    random.shuffle(keys)
    num_samples = len(keys)
    print('Exist {} patients'.format(num_samples))

    # split w.r.t keys
    train_keys = keys[:int(num_samples * (1 - val_ratio - test_ratio))]
    test_keys = keys[int(num_samples * (1 - val_ratio - test_ratio)):int(num_samples * (1 - val_ratio))]
    valid_keys = keys[int(num_samples * (1 - val_ratio)):]

    train_samples = list(itertools.chain.from_iterable([samples[x] for x in train_keys]))
    test_samples = list(itertools.chain.from_iterable([samples[x] for x in test_keys]))
    valid_samples = list(itertools.chain.from_iterable([samples[x] for x in valid_keys]))

    train_path = os.path.join(dst, 'train', str(tooth_id))
    test_path = os.path.join(dst, 'test', str(tooth_id))
    valid_path = os.path.join(dst, 'valid', str(tooth_id))

    for smp in train_samples:
        shutil.copytree(os.path.join(root, smp), os.path.join(train_path, smp))
    for smp in test_samples:
        shutil.copytree(os.path.join(root, smp), os.path.join(test_path, smp))
    for smp in valid_samples:
        shutil.copytree(os.path.join(root, smp), os.path.join(valid_path, smp))



def partition_data_good(root, dst, tooth_id, val_ratio=0.1, test_ratio=0.2):
    samples = get_tooth_set_good(root)
    keys = list(samples.keys())
    random.shuffle(keys)
    num_samples = len(keys)
    print('Exist {} patients'.format(num_samples))

    # split w.r.t keys
    train_keys = keys[:int(num_samples * (1 - val_ratio - test_ratio))]
    test_keys = keys[int(num_samples * (1 - val_ratio - test_ratio)):int(num_samples * (1 - val_ratio))]
    valid_keys = keys[int(num_samples * (1 - val_ratio)):]

    train_samples = list(itertools.chain.from_iterable([samples[x] for x in train_keys]))
    test_samples = list(itertools.chain.from_iterable([samples[x] for x in test_keys]))
    valid_samples = list(itertools.chain.from_iterable([samples[x] for x in valid_keys]))

    train_path = os.path.join(dst, 'train', str(tooth_id))
    test_path = os.path.join(dst, 'test', str(tooth_id))
    valid_path = os.path.join(dst, 'valid', str(tooth_id))

    for smp in train_samples:
        shutil.copytree(os.path.join(root, smp), os.path.join(train_path, smp))
    for smp in test_samples:
        shutil.copytree(os.path.join(root, smp), os.path.join(test_path, smp))
    for smp in valid_samples:
        shutil.copytree(os.path.join(root, smp), os.path.join(valid_path, smp))


def get_tooth_set_good(root):
    '''
    :param root: file folder
    :return: e.g., dictionary {'DC01002828426X02': ['DC01002828426X02_L01_36', 'DC01002828426X02_L01T_36', 'DC01002828426X02_L04_36']}
    '''
    folders = os.listdir(root)
    #pid = [(folder.split('_')[0], folder) for folder in folders if folder.startswith('DC')]
    pid = [(folder.split('_')[0], folder) for folder in folders if folder.startswith('C') and 'good' in folder]

    data_dict = {}
    for idx, folder in pid:
        if idx in data_dict.keys():
            data_dict[idx].append(folder)
        else:
            data_dict[idx] = [folder]

    return data_dict


def get_tooth_set(root):
    '''
    :param root: file folder
    :return: e.g., dictionary {'DC01002828426X02': ['DC01002828426X02_L01_36', 'DC01002828426X02_L01T_36', 'DC01002828426X02_L04_36']}
    '''
    folders = os.listdir(root)
    #pid = [(folder.split('_')[0], folder) for folder in folders if folder.startswith('DC')]
    pid = [(folder.split('_')[0], folder) for folder in folders if folder.startswith('C')]

    data_dict = {}
    for idx, folder in pid:
        if idx in data_dict.keys():
            data_dict[idx].append(folder)
        else:
            data_dict[idx] = [folder]

    return data_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HolePrediction')
    parser.add_argument('--data', type=str, default='../data/std_data',
                        help='path to xyz data')
    parser.add_argument('--save', type=str, default='../data/std_split_data',
                        help='path to exp data')
    parser.add_argument('--id', type=int, default=11,
                        help='tooth id')
    parser.add_argument('--all', type=bool, default=False,
                        help='Use all teeth or not')
    args = parser.parse_args()

    if not os.path.isdir(args.save):
        os.makedirs(args.save)
    if args.all:
        args.id = [11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 41, 42,
                   43, 44, 45, 46, 47, 48]
        for id in args.id:
            partition_data_good(os.path.join(args.data, str(id)), args.save, id)
    else:
        partition_data(os.path.join(args.data, str(args.id)), args.save, args.id)

