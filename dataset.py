""" train and test dataset

author baiyu
"""
import os
import sys
import pickle

import numpy
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class CIFAR100Train(Dataset):
    """cangjie952 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        #if transform is given, we transoform data using
        with open(os.path.join(path, 'train'), 'rb') as cangjie952:
            self.data = pickle.load(cangjie952, encoding='bytes')
        self.transform = transform

    def __len__(self):
        return len(self.data['fine_labels'.encode()])

    def __getitem__(self, index):
        label = self.data['fine_labels'.encode()][index]
        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        image = numpy.dstack((r, g, b))

        print(f"Label: {label}, {type(label)}")
        print(f"Image: {image.shape}, {type(image)}")

        if self.transform:
            image = self.transform(image)
        return label, image

class CIFAR100Test(Dataset):
    """cangjie952 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        with open(os.path.join(path, 'test'), 'rb') as cangjie952:
            self.data = pickle.load(cangjie952, encoding='bytes')
        self.transform = transform

    def __len__(self):
        return len(self.data['data'.encode()])

    def __getitem__(self, index):
        label = self.data['fine_labels'.encode()][index]
        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        image = numpy.dstack((r, g, b))

        if self.transform:
            image = self.transform(image)
        return label, image







class CANGJIE952Train(Dataset):
    """cangjie952 train dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        self.root_dir = path
        self.transform = transform
        self.labels = []
        self.image_paths = []
        self.data = []

        labels = sorted(os.listdir(self.root_dir), key=lambda x: int(x))
        for label in labels:
            if int(label) % 100 == 0:
                print(f"Train dataset: processing class {label}.")
            class_dir = os.path.join(self.root_dir, label)
            label_image_names = sorted(os.listdir(class_dir))
            label_image_paths = [os.path.join(class_dir, image_name) for image_name in label_image_names]
            if os.path.isdir(class_dir):            
                self.labels.extend(len(label_image_paths) * [label])  
                self.image_paths.extend(label_image_paths)
                for image_path in label_image_paths:
                    image = Image.open(image_path).convert('L').resize((32, 32))       # Convert to grayscale and resize to fit input tensor
                    image = np.array(image, dtype=np.float32)                   
                    image /= 255.0                          
                    self.data.append(image)

        self.data = np.stack(self.data)
        self.data = np.expand_dims(self.data, axis=-1)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):

        image = Image.open(self.image_paths[index]).convert('L').resize((32, 32))
        label = int(self.labels[index])

        if self.transform:
            image = self.transform(image)

        return image, label
    

class CANGJIE952Test(Dataset):
    """cangjie952 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        self.root_dir = path
        self.transform = transform
        self.labels = []
        self.image_paths = []
        self.data = []

        labels = sorted(os.listdir(self.root_dir), key=lambda x: int(x))
        for label in labels:
            if int(label) % 100 == 0:
                print(f"Test dataset: processing class {label}.")
            class_dir = os.path.join(self.root_dir, label)
            label_image_names = sorted(os.listdir(class_dir))
            label_image_paths = [os.path.join(class_dir, image_name) for image_name in label_image_names]
            if os.path.isdir(class_dir):            
                self.labels.extend(len(label_image_paths) * [label])  
                self.image_paths.extend(label_image_paths)
                for image_path in label_image_paths:
                    image = Image.open(image_path).convert('L').resize((32, 32))       # Convert to grayscale and resize to fit input tensor
                    image = np.array(image, dtype=np.float32)                   
                    image /= 255.0                          
                    self.data.append(image)

        self.data = np.stack(self.data)
        self.data = np.expand_dims(self.data, axis=-1)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):

        image = Image.open(self.image_paths[index]).convert('L').resize((32, 32))
        label = int(self.labels[index])

        if self.transform:
            image = self.transform(image)

        return image, label