import torch
import numpy as np
import torch.nn.functional as F
import os
import json
import scipy.io as spio
import hdf5storage

from torch.utils.data import Dataset, DataLoader

class rwcpDataset(Dataset):
    def __init__(self, X_data, Y_data, max_duration):
        self.x_data = X_data
        self.y_data = Y_data - 1
        self.max_duration = max_duration
        self.len = X_data.shape[0]

    def __getitem__(self, index):
        x = self.x_data[index, 0]
        len = x.shape[1]
        x = torch.from_numpy(x)
        x = F.pad(x, (0, self.max_duration - len))  # zero-pad all the feature to the max length
        y = torch.from_numpy(np.array(self.y_data[index]))

        return x, y

    def __len__(self):
        return self.len


def get_train_loader(data_dir, batch_size=16, num_workers=2, pin_memory=False):
    """
    Utility function for loading and returning train multi-process iterators over the dataset.

    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    """
    # Load Data
    data = spio.loadmat(data_dir)
    Xtr = data['FBETrainList']
    Ytr = np.squeeze(data['train_labels'])
    maxTrainLength = int(data['maxTrainLength'])

    train_loader_obj = rwcpDataset(Xtr, Ytr, maxTrainLength)

    train_loader = DataLoader(
        dataset=train_loader_obj, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader


def get_test_loader(data_dir,
                    batch_size=16,
                    num_workers=2,
                    pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the dataset.

    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - test_loader: testing set iterator.
    """
    # Load Data
    data = hdf5storage.loadmat(data_dir)
    Xte = data['FBETestList']
    Yte = np.squeeze(data['test_labels'])
    maxTestLength = int(data['maxTestLength'])

    test_loader_obj = rwcpDataset(Xte, Yte, maxTestLength)

    test_loader = DataLoader(
        dataset=test_loader_obj, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return test_loader


def dump_json(obj, fdir, name):
    """
    Dump python object in json
    """
    if fdir and not os.path.exists(fdir):
        os.makedirs(fdir)
    with open(os.path.join(fdir, name), "w") as f:
        json.dump(obj, f, indent=4, sort_keys=False)


def training_ann(model, trainloader, optimizer, criterion, device):
    model.train()  # Put the model in train mode

    running_loss = 0.0
    total = 0
    correct = 0
    for i_batch, (inputs, labels) in enumerate(trainloader, 1):
        # Transfer to GPU
        inputs, labels = inputs.type(torch.FloatTensor).to(device), \
                         labels.type(torch.LongTensor).to(device)

        # Model computation and weight update
        y_pred = model.forward(inputs)
        loss = criterion(y_pred, labels)
        _, predicted = torch.max(y_pred.data, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()

        epoch_loss = running_loss / i_batch
        acc_train = correct / total

    return model, acc_train, epoch_loss


def testing_ann(model, testLoader, criterion, device, maskProb=1.0):
    model.eval()  # Put the model in test mode

    running_loss = 0.0
    correct = 0
    total = 0
    for data in testLoader:
        inputs, labels = data

        # Transfer to GPU
        inputs, labels = inputs.type(torch.FloatTensor).to(device), \
                         labels.type(torch.LongTensor).to(device)

        prob = torch.ones_like(inputs)*maskProb
        random_mask = torch.bernoulli(prob)
        y_pred = model.forward(inputs*random_mask)

        # forward pass
        # y_pred = model.forward(inputs)
        loss = criterion(y_pred, labels)
        _, predicted = torch.max(y_pred.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()

    # calculate epoch statistics
    epoch_loss = running_loss / len(testLoader)
    acc = correct / total

    return acc, epoch_loss
