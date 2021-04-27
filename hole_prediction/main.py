from __future__ import print_function

import argparse
import json
import os
from shutil import copy2

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from data import HoleDataset, HoleDataset_bad
from loss import get_loss_criterion
from model import get_model
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

torch.backends.cudnn.enabled = False


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, **params):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.params = params

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=1)
        logpt = (1 - torch.exp(logpt)) ** self.gamma * logpt
        loss = F.nll_loss(logpt, target, **self.params)

        return loss


class Metric(object):
    def __init__(self):
        self.acc = 0
        self.loss = 0
        self.count = 0
        self.pos_acc, self.neg_acc, self.fp_acc, self.fn_acc = 0, 0, 0, 0

    def update(self, y, y_, criterion):
        self.acc += (y_.argmax(dim=1) == y).to(torch.float32).mean().item()
        self.loss += criterion(y_, y).item()
        self.count += 1

        y_ = y_.argmax(dim=1).detach().cpu().numpy()
        y = y.cpu().numpy()
        self.pos_acc += (np.mean(y_[y == 1] == 1))
        self.neg_acc += (np.mean(y_[y == 0] == 0))
        self.fp_acc += (np.mean(y_[y == 0] == 1))
        self.fn_acc += (np.mean(y_[y == 1] == 0))

    def avg_loss(self):
        return self.loss / self.count

    def avg_acc(self):
        return self.acc / self.count

    def avg_stat(self):
        return self.pos_acc / self.count, self.neg_acc / self.count, self.fp_acc / self.count, self.fn_acc / self.count


def init():
    model_path = os.path.join('checkpoints', args.exp_name, 'models')
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    src_path = os.path.join('checkpoints', args.exp_name, 'src')
    if not os.path.isdir(src_path):
        os.makedirs(src_path)
    # copy2('data.py', src_path)
    # copy2('model.py', src_path)
    # copy2('main.py', src_path)


def train(args):
    train_loader = DataLoader(HoleDataset('../data/std_split_data/train', args.tooth_ids), num_workers=0,batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(HoleDataset('../data/std_split_data/valid', args.tooth_ids, train=False), num_workers=0,batch_size=4, shuffle=False, drop_last=False)
    # Try to load models
    device = torch.device("cuda")
    model = get_model(args.model, args).to(device)
    print(str(model))
    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'weighted_ce':
        with open(args.stat) as json_file:
            stat = json.load(json_file)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1 / stat['neg_rate'], 1 / stat['pos_rate']])).cuda()
    elif args.loss == 'weighted_ce_var_i':
        with open(args.stat) as json_file:
            stat = json.load(json_file)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1 - stat['neg_rate'], 1 - stat['pos_rate']])).cuda()
    elif args.loss == 'weighted_ce_var_ii':
        with open(args.stat) as json_file:
            stat = json.load(json_file)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([stat['pos_rate'] / stat['neg_rate'], stat['pos_rate'] / stat['pos_rate']])).cuda()
    elif args.loss == 'weighted_ce_var_iii':
        with open(args.stat) as json_file:
            stat = json.load(json_file)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([stat['neg_rate'] / stat['neg_rate'], stat['neg_rate'] / stat['pos_rate']])).cuda()
    elif args.loss == 'log_weighted_ce':
        with open(args.stat) as json_file:
            stat = json.load(json_file)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([np.log(1.1 + 1 / stat['neg_rate']),
                                                             np.log(1.1 + 1 / stat['pos_rate'])])).cuda()
    elif args.loss == 'focal_loss':
        if args.alpha == 0.5:
            criterion = FocalLoss(gamma=args.gamma)
        else:
            criterion = FocalLoss(gamma=args.gamma, weight=torch.tensor([args.alpha, 1 - args.alpha]).cuda())
    else:
        criterion = get_loss_criterion(name=args.loss, stats=args.stat, beta=args.beta)

    if args.opt == 'sgd':
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    elif args.opt == 'adam':
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        raise Exception("Not implemented")
    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)

    writer = SummaryWriter(os.path.join('runs', args.exp_name))

    best_test_acc = -1
    for epoch in tqdm.tqdm(range(args.epochs)):
        ####################
        # Train
        ####################
        model.train()
        train_metric = Metric()
        for x, y, _ in tqdm.tqdm(train_loader, leave=False):
            x, y = x.to(device).permute(0, 2, 1), y.to(device)

            opt.zero_grad()
            y_ = model(x)
            loss = criterion(y_, y)
            loss.backward()
            opt.step()

            train_metric.update(y, y_, criterion)
        scheduler.step()
        ####################
        # Test
        ####################
        if epoch % 10 == 0:
            model.eval()
            test_metric = Metric()
            with torch.no_grad():
                for idx, (x, y, _) in enumerate(test_loader):
                    x, y = x.to(device).permute(0, 2, 1), y.to(device)

                    y_ = model(x)

                    test_metric.update(y, y_, criterion)
                    if idx == 0:
                        x = x.permute(0, 2, 1)[:1]
                        color = torch.tensor([[[255, 0, 0]]]).to(device, torch.long).repeat(1, x.size(1), 1)
                        writer.add_mesh('y', x, colors=color * y[:1].unsqueeze(-1), global_step=epoch)
                        writer.add_mesh('y_', x, colors=color * y_[:1].argmax(dim=1).unsqueeze(-1), global_step=epoch)

            if test_metric.avg_acc() > best_test_acc:
                best_test_acc = test_metric.avg_acc()
                torch.save(model.state_dict(), os.path.join('checkpoints', args.exp_name, 'models/model.t7'))
        torch.save(model.state_dict(), os.path.join('checkpoints', args.exp_name, 'models/latest.t7'))
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join('checkpoints', args.exp_name, 'models/' + str(epoch) + '.t7'))
        ####################
        # Writer
        ####################
        writer.add_scalar('avg_train_loss', train_metric.avg_loss(), epoch)
        writer.add_scalar('avg_train_acc', train_metric.avg_acc(), epoch)
        writer.add_scalar('avg_test_loss', test_metric.avg_loss(), epoch)
        writer.add_scalar('avg_test_acc', test_metric.avg_acc(), epoch)

        avg_pos_acc, avg_neg_acc, avg_fp_acc, avg_fn_acc = train_metric.avg_stat()
        writer.add_scalar('avg_pos_acc', avg_pos_acc, epoch)
        writer.add_scalar('avg_neg_acc', avg_neg_acc, epoch)
        writer.add_scalar('avg_fp_acc', avg_fp_acc, epoch)
        writer.add_scalar('avg_fn_acc', avg_fn_acc, epoch)

        avg_pos_acc_t, avg_neg_acc_t, avg_fp_acc_t, avg_fn_acc_t = test_metric.avg_stat()
        writer.add_scalar('test_avg_pos_acc', avg_pos_acc_t, epoch)
        writer.add_scalar('test_avg_neg_acc', avg_neg_acc_t, epoch)

        print('Epoch {}: Pos_acc: {}, Neg_acc: {}'.format(epoch, avg_pos_acc, avg_neg_acc))

    writer.close()


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='HolePrediction')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', choices=['dgcnn'],
                        help='Model to use, [dgcnn]')
    parser.add_argument('--k', type=int, default=20,
                        help='Num of nearest neighbors to use')
    parser.add_argument('--loss', type=str, default='ce',
                        choices=['ce', 'weighted_ce', 'log_weighted_ce', 'weighted_ce_var_i', 'weighted_ce_var_ii',
                                 'weighted_ce_var_iii', 'focal_loss', 'dice_loss', 'weight_dice_loss', 'wce_dice_loss'],
                        help='Loss to use, [ce, weighted_ce, log_weighted_ce, weighted_ce_var_i, weighted_ce_var_ii, weighted_ce_var_iii, focal_loss, dice_loss, gen_dice_loss, wce_gdl_loss]')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of episode to train ')
    parser.add_argument('--p', type=float, default=0.4,
                        help='dropout ratio ')
    parser.add_argument('--opt', type=str, default='sgd', choices=['sgd', 'adam'],
                        help='Optimizer to use, [sgd, adam]')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--tooth_ids', type=str, default=None, help='tooth ids')
    parser.add_argument('--all', type=bool, default=False,
                        help='Use all teeth or not')
    parser.add_argument('--stat', type=str, default=None, help='data stat')
    parser.add_argument('--gamma', type=float, default=2, help='focal loss: gamma')
    parser.add_argument('--alpha', type=float, default=0.5, help='focal loss: alpha')
    parser.add_argument('--beta', type=float, default=1.0, help='wce dice loss weights')
    args = parser.parse_args()

    if args.all:
        args.exp_name = '_'.join(['{}[{}]'.format(k, v) for k, v in args.__dict__.items() if v != None])
        args.tooth_ids = [11,12,13,14,15,16,17,21,22,23,24,25,26,27,28,31,32,33,34,35,36,37,38,41,42,43,44,45,46,47,48]
    else:
        args.tooth_ids = [int(i) for i in args.tooth_ids.split(',')]
        args.exp_name = '_'.join(['{}[{}]'.format(k, v) for k, v in args.__dict__.items() if v != None])
    print(args)

    init()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train(args)
