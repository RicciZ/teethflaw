'''
Loss functions for segmentation
1: Cross entropy and weighted cross entropy
2: Focal loss
3: Overlap measures: Dice loss, Tversky Loss, Boundary loss
References
all Losses: Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations
Dice loss: V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
Tversky loss:  Tversky loss function for image segmentation using 3D fully convolutional deep networks
Boundary loss:  Boundary loss for highly unbalanced segmentation.
'''
import json

import torch
from torch import nn as nn
from torch.autograd import Variable


class DiceLoss(nn.Module):
    """Computes Dice Loss, as in https://arxiv.org/pdf/1707.03237.pdf Section 2: Dice Loss
    Additionally allows per-class weights to be provided.
    """

    def __init__(self, epsilon=1e-5, weight=None, ignore_index=None, sigmoid_normalization=False):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify sigmoid_normalization=False.
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)
        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False)
        else:
            weight = None

        per_channel_dice = compute_per_channel_dice(input, target, epsilon=self.epsilon, ignore_index=self.ignore_index,
                                                    weight=weight)
        # Average the Dice score across all channels/classes
        return torch.mean(1. - per_channel_dice)



class diceloss(nn.Module):
    def __init__(self, smooth=1e-4, weight=None):
        super(diceloss, self).__init__()

        self.smooth = smooth
        self.norm = nn.Softmax(dim=1)

    def forward(self, pred, target):
        pred = self.norm(pred) # pred.size = N * 2 * numpoints
        target = torch.nn.functional.one_hot(target, 2).transpose(1, 2)
        #assert pred.size() == target.size()
        iflat = pred.contiguous().view(-1)
        tflat = target.float().contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        return 1 - ((2. * intersection) /
                    (iflat.sum() + tflat.sum() + self.smooth))

class weight_diceloss(nn.Module):

    def __init__(self, smooth=1e-4, weight=None):
        super(weight_diceloss, self).__init__()

        self.smooth = smooth
        self.norm = nn.Softmax(dim=1)
        self.weight = weight

    def forward(self, pred, target, n_classes=2):
        pred = self.norm(pred)  # pred.size = N * 2 * numpoints
        target = torch.nn.functional.one_hot(target, 2).transpose(1, 2)
        loss = torch.tensor(0.0).cuda()
        for c in range(n_classes):
            iflat = pred[:, c].contiguous().view(-1)
            tflat = target[:, c].float().contiguous().view(-1)
            intersection = (iflat * tflat).sum()
            w = self.weight[c].cuda()
            loss += w * (1 - ((2. * intersection) /
                              (iflat.sum() + tflat.sum() + self.smooth)))
        return loss


class GeneralizedDiceLoss(nn.Module):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, epsilon=1e-5, weight=None, ignore_index=None, sigmoid_normalization=False):
        super(GeneralizedDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)
        target = torch.nn.functional.one_hot(target, 2).transpose(1, 2)
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"
        # mask ignore_index if present
        if self.ignore_index is not None:
            mask = target.clone().ne_(self.ignore_index)
            mask.requires_grad = False

            input = input * mask
            target = target * mask

        input = flatten(input)
        target = flatten(target)

        target = target.float()
        target_sum = target.sum(-1)
        class_weights = Variable(1. / (target_sum * target_sum).clamp(min=self.epsilon), requires_grad=False)

        intersect = (input * target).sum(-1) * class_weights
        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False)
            intersect = weight * intersect
        intersect = intersect.sum()

        denominator = ((input + target).sum(-1) * class_weights).sum()

        return 1. - 2. * intersect / denominator.clamp(min=self.epsilon)


class WCE_GDL_loss(nn.Module):
    def __init__(self, weight_ce, weight_dc=None, beta=1.0):
        super(WCE_GDL_loss, self).__init__()
        self.beta = beta
        self.ce = nn.CrossEntropyLoss(weight_ce)
        self.dc = GeneralizedDiceLoss(weight=weight_dc)

    def forward(self, pred, target):
        gdl_loss = self.dc(pred, target)
        wce_loss = self.ce(pred, target)
        result = wce_loss + self.beta * gdl_loss
        return result

class WCE_dice_loss(nn.Module):
    # weighted cross entropy loss with number of faces
    def __init__(self, weight_ce, smooth=1e-4, beta=1.0):
        super(WCE_dice_loss, self).__init__()
        self.beta = beta
        self.ce = nn.CrossEntropyLoss(weight_ce)
        self.dc = weight_diceloss(weight=weight_ce)

    def forward(self, pred, target):
        dice_loss = self.dc(pred, target)
        wce_loss = self.ce(pred, target)
        result = wce_loss + self.beta * dice_loss
        return result


class BDLoss(nn.Module):
    def __init__(self):
        """
        compute boudary loss
        only compute the loss of foreground
        ref: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L74
        """
        super(BDLoss, self).__init__()

    def forward(self, net_output, bound):
        """
        net_output: (batch_size, class, L)
        target: ground truth, shape: (batch_size, 1, L)
        bound: precomputed distance map, shape (batch_size, class, L)
        """
        net_output = softmax_helper(net_output)
        pc = net_output[:, 1:, ...].type(torch.float32)
        dc = bound[:, 1:, ...].type(torch.float32)

        multipled = torch.einsum("bcxyz,bcxyz->bcxyz", pc, dc)
        bd_loss = multipled.mean()

        return bd_loss


def softmax_helper(x):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/nd_softmax.py
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)


def compute_per_channel_dice(input, target, epsilon=1e-5, ignore_index=None, weight=None):
    # assumes that input is a normalized probability
    # input and target shapes must match
    target = torch.nn.functional.one_hot(target, 2).transpose(1, 2)
    assert input.size() == target.size(), "'input' and 'target' must have the same shape: N * 2 * L"

    # mask ignore_index if present
    if ignore_index is not None:
        mask = target.clone().ne_(ignore_index)
        mask.requires_grad = False

        input = input * mask
        target = target * mask

    input = flatten(input)
    target = flatten(target)

    target = target.float()
    # Compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    denominator = (input + target).sum(-1)
    return 2. * intersect / denominator.clamp(min=epsilon)


def flatten(x):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, L) -> (C, N * L)
       in our case, C = 2
    """
    C = x.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, x.dim()))
    # Transpose: (N, C, L) -> (C, N, L)
    transposed = x.permute(axis_order)
    # Flatten: (C, N, L) -> (C, N * L)
    return transposed.contiguous().view(C, -1)


def get_loss_criterion(name, stats=None, beta=1.0):
    if name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss(weight=None).cuda()
    elif name == 'gen_dice_loss':
        return GeneralizedDiceLoss(weight=None).cuda()
    elif name == 'dice_loss':
        return diceloss().cuda()
    elif name == 'weight_dice_loss':
        with open(stats) as json_file:
            stat = json.load(json_file)
        return weight_diceloss(weight=torch.tensor([1 - stat['neg_rate'], 1 - stat['pos_rate']])).cuda()
    elif name == 'wce_dice_loss':
        with open(stats) as json_file:
            stat = json.load(json_file)
        return WCE_dice_loss(weight_ce=torch.tensor([1 - stat['neg_rate'], 1 - stat['pos_rate']]), beta=beta).cuda()
    else:
        raise RuntimeError("Unsupported loss function: {}.".format(name))
