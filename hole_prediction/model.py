import torch
import torch.nn as nn


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    return pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)


def get_graph_feature(x, k=20, idx=None):
    batch_size, num_dims, num_points = x.size()
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)

    idx = idx + torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)

    return feature


class DGCNN(nn.Module):
    def __init__(self, k):
        super(DGCNN, self).__init__()
        self.k = k

        self.conv1 = nn.Sequential(nn.Conv2d(15 * 2, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.deconv5 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                     nn.BatchNorm1d(128),
                                     nn.LeakyReLU(negative_slope=0.2))
        self.deconv4 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1, bias=False),
                                     nn.BatchNorm1d(64),
                                     nn.LeakyReLU(negative_slope=0.2))
        self.output = nn.Conv1d(64, 2, kernel_size=1, bias=False)

    def forward(self, x):
        x = get_graph_feature(x, k=self.k)          # x(b,d=15,n,k) -> x(b,2*d=30,n,k)
        x = self.conv1(x)                           # x(b,30,n,k) -> x(b,64,n,k)
        x1 = x.max(dim=-1, keepdim=False)[0]        # x(b,64,n,k) -> x1(b,64,n)

        x = get_graph_feature(x1, k=self.k)         # x1(b,64,n) -> x(b,2*64,n,k)
        x = self.conv2(x)                           # x(b,2*64,n,k) -> x(b,64,n,k)
        x2 = x.max(dim=-1, keepdim=False)[0]        # x(b,64,n,k) -> x2(b,64,n)

        x = get_graph_feature(x2, k=self.k)         # x2(b,64,n) -> x(b,2*64,n,k)
        x = self.conv3(x)                           # x(b,2*64,n,k) -> x(b,128,n,k)
        x3 = x.max(dim=-1, keepdim=False)[0]        # x(b,128,n,k) -> x3(b,128,n)

        x = get_graph_feature(x3, k=self.k)         # x3(b,128,n) -> x(b,2*128,n,k)
        x = self.conv4(x)                           # x(b,2*128,n,k) -> x(b,256,n,k)
        x4 = x.max(dim=-1, keepdim=False)[0]        # x(b,256,n,k) -> x4(b,256,n)

        x = torch.cat((x1, x2, x3, x4), dim=1)      # x1,x2,x3,x4 -> x(b,64+64+128+256=512,n)
        x = self.conv5(x)                           # x(b,512,n) -> x(b,256,n)
        x = self.deconv5(x)                         # x(b,256,n) -> x(b,128,n)
        x = self.deconv4(x)                         # x(b,128,n) -> x(b,64,n)

        return self.output(x)                       # x(b,64,n) -> x(b,2,n)


class SkipDGCNN(DGCNN):
    def __init__(self, k):
        super(SkipDGCNN, self).__init__(k=k)

    def forward(self, x):
        return super(SkipDGCNN, self).forward(x) + x


def get_model(name, args):
    if name == 'dgcnn':
        return DGCNN(args.k)
    elif name == 'skip_dgcnn':
        return SkipDGCNN(args.k)
    else:
        raise Exception("Not implemented")
