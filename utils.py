""" helper function

author baiyu
"""
import os
import sys
import re
import datetime

import numpy as np

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import CANGJIE952, CIFAR100Test, CIFAR100Train
from conf import settings
import editdistance


def get_network(args):
    """ return given network
    """

    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif args.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif args.net == 'xception':
        from models.xception import xception
        net = xception()
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif args.net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif args.net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif args.net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif args.net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif args.net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif args.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif args.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif args.net == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif args.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif args.net == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34()
    elif args.net == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50()
    elif args.net == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101()
    elif args.net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()
    elif args.net == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet()
    elif args.net == 'stochasticdepth18':
        from models.stochasticdepth import stochastic_depth_resnet18
        net = stochastic_depth_resnet18()
    elif args.net == 'stochasticdepth34':
        from models.stochasticdepth import stochastic_depth_resnet34
        net = stochastic_depth_resnet34()
    elif args.net == 'stochasticdepth50':
        from models.stochasticdepth import stochastic_depth_resnet50
        net = stochastic_depth_resnet50()
    elif args.net == 'stochasticdepth101':
        from models.stochasticdepth import stochastic_depth_resnet101
        net = stochastic_depth_resnet101()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cangjie952 training dataset
        std: std of cangjie952 training dataset
        path: path to cangjie952 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    cangjie952_training = CANGJIE952('train', settings.CANGJIE952_TRAIN_PATH, settings.CANGJIE952_LABEL_PATH, transform=transform_train)
    cangjie952_training_loader = DataLoader(
        cangjie952_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cangjie952_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cangjie952 test dataset
        std: std of cangjie952 test dataset
        path: path to cangjie952 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cangjie952_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    cangjie952_test = CANGJIE952('test', settings.CANGJIE952_VAL_PATH, settings.CANGJIE952_LABEL_PATH, transform=transform_test)
    cangjie952_test_loader = DataLoader(
        cangjie952_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cangjie952_test_loader

def compute_mean_std(cangjie952_dataset):
    """compute the mean and std of cangjie952 dataset
    Args:
        cangjie952_training_dataset or cangjie952_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    mean = cangjie952_dataset.data.mean().item()
    std = cangjie952_dataset.data.std().item()

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]

def one_hot_encoding_to_cangjie_encoding(one_hot_encoding: torch.Tensor, type: str) -> list[str]:
    cangjie_encoding = []
    THRESHOLD = 0.8
    AMPLIFY = 1000

    if type == 'ground_truth':        
        for one_hot in one_hot_encoding:
            if one_hot.sum().item() == 0.0:
                cangjie_encoding.append('zc')
            else:
                cangjie_str = ''
                for i in range(5):
                    chunk = one_hot[i * 26: (i + 1) * 26]
                    if chunk.sum().item() == 0.0:
                        break
                    latin_char = chr(chunk.argmax().item() + ord('a'))
                    cangjie_str += latin_char
                cangjie_encoding.append(cangjie_str)
    else:
        for one_hot in one_hot_encoding:
            cangjie_str = ''
            for i in range(5):
                chunk = one_hot[i * 26: (i + 1) * 26] * AMPLIFY
                # convert chunk to softmax, and select the max, but only convert it to latin char when the softmax value is above THRESHOLD
                chunk = torch.nn.functional.softmax(chunk, dim=0)
                if chunk.max().item() < THRESHOLD:
                    break
                latin_char = chr(chunk.argmax().item() + ord('a'))
                cangjie_str += latin_char
            cangjie_encoding.append(cangjie_str)

    return cangjie_encoding


def get_prediction_error(preds, ground_truths):
    total_length = 0
    total_distance = 0
    
    for id in range(len(ground_truths)):
        pred = preds[id]
        ground_truth = ground_truths[id]
        if ground_truth == 'zc':
            total_length += 1
            total_distance += 0 if pred.startswith('zc') else 1
        else:
            total_length += len(ground_truth)
            total_distance += editdistance.eval(pred, ground_truth)

    return total_distance, total_length



if __name__ == "__main__":

    # print("CIFAR100_train...")
    # cifar100_train = torchvision.datasets.CIFAR100(root='./data', train=True, download=True)
    # print(cifar100_train.data.shape)
    # print(type(cifar100_train.data))
    # print(type(cifar100_train.data[0]))
    # print(cifar100_train[0])
    
    cangjie952_train = CANGJIE952('train', settings.CANGJIE952_TRAIN_PATH, settings.CANGJIE952_LABEL_PATH)
    train_mean, train_std = compute_mean_std(cangjie952_train)
    print(f"Train mean: {train_mean}, train std: {train_std}.")

    cangjie952_test = CANGJIE952('test', settings.CANGJIE952_VAL_PATH, settings.CANGJIE952_LABEL_PATH)
    test_mean, test_std = compute_mean_std(cangjie952_test)
    print(f"Test mean: {test_mean}, test std: {test_std}.")

