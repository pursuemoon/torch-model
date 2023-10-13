# -- coding: utf-8 --

import torch
import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import MNIST

from utils.log import logger
from utils.time import human_readable_time
from base_model import BaseModel

def get_dataloader(batch_size, is_train, is_shuffle):
    trans_func = Compose([
        ToTensor(),
    ])
    dataset = MNIST(root='./data', train=is_train, download=True, transform=trans_func)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=is_shuffle)

class MnistCnnModel(BaseModel):
    def __init__(self):
        super(MnistCnnModel, self).__init__()
        self.cv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, stride=1, padding=1)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(10*7*7, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view([-1, 1, 28, 28])
        x = F.relu(self.cv1(x))
        x = self.mp1(x)

        x = x.view([-1, 5, 14, 14])
        x = F.relu(self.cv2(x))
        x = self.mp2(x)

        x = x.view([-1, 10*7*7])
        x = F.relu(self.fc1(x))

        x = self.output(x)
        return x

def train(model:MnistCnnModel, max_epoch, learning_rate, batch_size, device):
    start_time = time.time()

    model.to(device)
    data_loader = get_dataloader(batch_size, True, True)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-7)
    loss_func = nn.CrossEntropyLoss()

    total_cnt = max_epoch * len(data_loader.dataset)
    process_bar = tqdm.tqdm(total=total_cnt, colour='green', ncols=120, unit_scale=True, desc="Training")

    for i in range(max_epoch):
        if model.is_being_stoped:
            break
        for idx, (input, target) in enumerate(data_loader):
            input, target = input.to(device), target.to(device)
            optimizer.zero_grad()

            output = model(input)
            loss = loss_func(output, target)
            loss.backward()

            optimizer.step()

            if idx % 1000 ==0:
                logger.debug('epoch={}, idx={}, loss={}'.format(i, idx, loss))
            process_bar.update(len(target))

    process_bar.close()

    model.is_trained = True

    end_time = time.time()
    logger.info('Training ended. Time used: {}'.format(human_readable_time(end_time - start_time)))

def test(model, batch_size, device):
    model.to(device)
    data_loader = get_dataloader(batch_size, False, False)

    correct_cnt = 0
    for idx, (input, target) in enumerate(data_loader):
        input, target = input.to(device), target.to(device)
        output = model(input)
        pred = output.max(dim=-1)[-1]

        correct_cnt += pred.eq(target).int().sum()
    accuracy = correct_cnt / len(data_loader.dataset)
    logger.info('accuracy={}'.format(accuracy))

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    path = 'models/mnist-cnn'

    # model = MnistCnnModel()
    # train(model, 50, 0.001, 20, device)
    # torch.save(model, path)

    model = torch.load(path)

    test(model, 100, device)
