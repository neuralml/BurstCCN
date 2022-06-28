import sys

import math
import random

import torch
import torchvision
from torchvision import datasets
import torch.utils.data as data

from modules.networks import ANN


import numpy as np
# datasets.MNIST.resources = [
#         (
#         'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
#         (
#         'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
#         ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
#         ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')
#     ]


def get_mnist_dataset(data_dir, train_batch_size, test_batch_size=1000, use_validation=False, train_subset_size=None):
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    full_train_set = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform_train)

    if use_validation:
        train_set, validation_set = torch.utils.data.random_split(full_train_set, [50000, 10000])
    else:
        train_set, validation_set = full_train_set, None

        # train_set, validation_set = torch.utils.data.random_split(full_train_set, [5000, 55000])[0], None

    test_set = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform_test)

    kwargs = {'num_workers': 0, 'pin_memory': True}

    if train_subset_size is not None:
        train_set.data = train_set.data[:train_subset_size]

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True, **kwargs)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=test_batch_size, shuffle=False, **kwargs) if validation_set is not None else None
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False, **kwargs)

    return train_loader, validation_loader, test_loader


def get_cifar10_dataset(data_dir, train_batch_size, test_batch_size=1000, use_validation=False):

    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    full_train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)

    if use_validation:
        train_set, validation_set = torch.utils.data.random_split(full_train_set, [40000, 10000])
    else:
        train_set, validation_set = full_train_set, None

    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    kwargs = {'num_workers': 0, 'pin_memory': True}

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True, **kwargs)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=test_batch_size, shuffle=False, **kwargs) if validation_set is not None else None
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False, **kwargs)

    return train_loader, validation_loader, test_loader


def get_xor_dataset():
    X_train = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    # Y_train = torch.Tensor([0, 1, 1, 0]).view(-1, 1).to(torch.int64)
    Y_train = torch.Tensor([0, 1, 1, 0]).view(-1, 1)#.to(torch.int64)

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)

    return train_loader, train_loader, train_loader


class ContinuousDataLoader:
    def __init__(self, num_inputs, batch_size, shuffle, complexity, device):
        self.num_inputs = num_inputs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

        min_period = 40.0
        self.cosine_freqs = [random.random() * 2*math.pi/min_period for i in range(self.num_inputs)]
        self.cosine_phases = [random.random() * (math.pi * 2.0) for i in range(self.num_inputs)]

        self.timestep = 0
        self.ann = ANN(num_inputs, complexity, 25, 1, device)

        for layer in self.ann.linear_layers:
            layer.weight.data *= 3

    def __iter__(self):
        return self

    def __next__(self):

        if self.shuffle:
            timestep = random.randint(0, sys.maxsize)
        else:
            timestep = self.timestep
            self.timestep += 1

        inputs = torch.tensor([math.cos(freq*timestep + phase) for freq, phase in zip(self.cosine_freqs, self.cosine_phases)], device=self.device)
        targets = self.ann(inputs).detach()

        return inputs, targets


def get_continuous_dataset(num_inputs, batch_size, shuffle, complexity, device):
    return ContinuousDataLoader(num_inputs=num_inputs, batch_size=batch_size, shuffle=shuffle, complexity=complexity, device=device)


if __name__ == '__main__':
    num_inputs = 3
    continuous_dataset = get_continuous_dataset(num_inputs, 1, False, 1, 'cuda')

    inputs_list = []
    target_list = []
    for i, (inputs, targets) in enumerate(continuous_dataset):
        # print(inputs, targets)
        inputs_list.append(inputs.cpu())
        target_list.append(targets.cpu())
        if i == 600:
            break


    inputs = torch.vstack(inputs_list)
    targets = torch.vstack(target_list)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(num_inputs, 2, figsize=(10, 10))

    for i in range(num_inputs):
        ax[i, 0].plot(range(len(inputs)), inputs[:, i])

    ax[0, 1].plot(range(len(targets)), targets[:, 0])

    fig.savefig('sins.pdf', bbox_inches='tight', dpi=600)

    plt.show()
