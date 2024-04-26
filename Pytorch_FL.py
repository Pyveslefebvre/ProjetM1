import os
import random
from tqdm import tqdm
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import Compose
torch.backends.cudnn.benchmark=True

# Hyperparametre
classes_pc = 2
num_clients = 20
num_selected = 6
num_rounds = 150
epochs = 5
batch_size = 32
baseline_num = 100
retrain_epochs = 20


#### Charger le dataset cifar et le transformer en dataset non-IID -> ce qui ressemble au monde réel

def get_cifar10():
    '''Return CIFAR10 train/test data and labels as numpy arrays'''
    data_train = torchvision.datasets.CIFAR10('./data', train=True, download=True)
    data_test = torchvision.datasets.CIFAR10('./data', train=False, download=True)

    x_train, y_train = data_train.data.transpose((0, 3, 1, 2)), np.array(data_train.targets)
    x_test, y_test = data_test.data.transpose((0, 3, 1, 2)), np.array(data_test.targets)

    return x_train, y_train, x_test, y_test


def print_image_data_stats(data_train, labels_train, data_test, labels_test):
    print("\nData: ")
    print(" - Train Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
        data_train.shape, labels_train.shape, np.min(data_train), np.max(data_train),
        np.min(labels_train), np.max(labels_train)))
    print(" - Test Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
        data_test.shape, labels_test.shape, np.min(data_train), np.max(data_train),
        np.min(labels_test), np.max(labels_test)))


def clients_rand(train_len, nclients):
    '''
    train_len: size of the train data
    nclients: number of clients

    Returns: to_ret

    This function creates a random distribution
    for the clients, i.e. number of images each client
    possess.
    '''
    client_tmp = []
    sum_ = 0
    #### creating random values for each client ####
    for i in range(nclients - 1):
        tmp = random.randint(10, 100)
        sum_ += tmp
        client_tmp.append(tmp)

    client_tmp = np.array(client_tmp)
    #### using those random values as weights ####
    clients_dist = ((client_tmp / sum_) * train_len).astype(int)
    num = train_len - clients_dist.sum()
    to_ret = list(clients_dist)
    to_ret.append(num)
    return to_ret


def split_image_data(data, labels, n_clients=100, classes_per_client=10, shuffle=True, verbose=True):
    '''
    Splits (data, labels) among 'n_clients s.t. every client can holds 'classes_per_client' number of classes
    Input:
      data : [n_data x shape]
      labels : [n_data (x 1)] from 0 to n_labels
      n_clients : number of clients
      classes_per_client : number of classes per client
      shuffle : True/False => True for shuffling the dataset, False otherwise
      verbose : True/False => True for printing some info, False otherwise
    Output:
      clients_split : client data into desired format
    '''
    #### constants ####
    n_data = data.shape[0]
    n_labels = np.max(labels) + 1

    ### client distribution ####
    data_per_client = clients_rand(len(data), n_clients)
    data_per_client_per_class = [np.maximum(1, nd // classes_per_client) for nd in data_per_client]

    # sort for labels
    data_idcs = [[] for i in range(n_labels)]
    for j, label in enumerate(labels):
        data_idcs[label] += [j]
    if shuffle:
        for idcs in data_idcs:
            np.random.shuffle(idcs)

    # split data among clients
    clients_split = []
    c = 0
    for i in range(n_clients):
        client_idcs = []

        budget = data_per_client[i]
        c = np.random.randint(n_labels)
        while budget > 0:
            take = min(data_per_client_per_class[i], len(data_idcs[c]), budget)

            client_idcs += data_idcs[c][:take]
            data_idcs[c] = data_idcs[c][take:]

            budget -= take
            c = (c + 1) % n_labels

        clients_split += [(data[client_idcs], labels[client_idcs])]

    def print_split(clients_split):
        print("Data split:")
        for i, client in enumerate(clients_split):
            split = np.sum(client[1].reshape(1, -1) == np.arange(n_labels).reshape(-1, 1), axis=1)
            print(" - Client {}: {}".format(i, split))
        print()

        if verbose:
            print_split(clients_split)

    clients_split = np.array(clients_split)

    return clients_split


def shuffle_list(data):
    '''
    This function returns the shuffled data
    '''
    for i in range(len(data)):
        tmp_len = len(data[i][0])
        index = [i for i in range(tmp_len)]
        random.shuffle(index)
        data[i][0], data[i][1] = shuffle_list_data(data[i][0], data[i][1])
    return data


def shuffle_list_data(x, y):
    '''
    This function is a helper function, shuffles an
    array while maintaining the mapping between x and y
    '''
    inds = list(range(len(x)))
    random.shuffle(inds)
    return x[inds], y[inds]


class CustomImageDataset(Dataset):
    '''
    A custom Dataset class for images
    inputs : numpy array [n_data x shape]
    labels : numpy array [n_data (x 1)]
    '''

    def __init__(self, inputs, labels, transforms=None):
        assert inputs.shape[0] == labels.shape[0]
        self.inputs = torch.Tensor(inputs)
        self.labels = torch.Tensor(labels).long()
        self.transforms = transforms

    def __getitem__(self, index):
        img, label = self.inputs[index], self.labels[index]

        if self.transforms is not None:
            img = self.transforms(img)

        return (img, label)

    def __len__(self):
        return self.inputs.shape[0]


def get_default_data_transforms(train=True, verbose=True):
    transforms_train = {
        'cifar10': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
        # (0.24703223, 0.24348513, 0.26158784)
    }
    transforms_eval = {
        'cifar10': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    }
    if verbose:
        print("\nData preprocessing: ")
        for transformation in transforms_train['cifar10'].transforms:
            print(' -', transformation)
        print()

    return (transforms_train['cifar10'], transforms_eval['cifar10'])


def get_data_loaders(nclients, batch_size, classes_pc=10, verbose=True):
    x_train, y_train, x_test, y_test = get_cifar10()

    if verbose:
        print_image_data_stats(x_train, y_train, x_test, y_test)

    transforms_train, transforms_eval = get_default_data_transforms(verbose=False)

    split = split_image_data(x_train, y_train, n_clients=nclients,
                             classes_per_client=classes_pc, verbose=verbose)

    split_tmp = shuffle_list(split)

    client_loaders = [torch.utils.data.DataLoader(CustomImageDataset(x, y, transforms_train),
                                                  batch_size=batch_size, shuffle=True) for x, y in split_tmp]

    test_loader = torch.utils.data.DataLoader(CustomImageDataset(x_test, y_test, transforms_eval), batch_size=100,
                                              shuffle=False)

    return client_loaders, test_loader

### Creation du réseau de neuronne

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        output = F.log_softmax(out, dim=1)
        return output

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

# Creation d'une base de donnée à envoyer au client

def baseline_data(num):
  '''
  Returns baseline data loader to be used on retraining on global server
  Input:
        num : size of baseline data
  Output:
        loader: baseline data loader
  '''
  xtrain, ytrain, xtmp,ytmp = get_cifar10()
  x , y = shuffle_list_data(xtrain, ytrain)

  x, y = x[:num], y[:num]
  transform, _ = get_default_data_transforms(train=True, verbose=False)
  loader = torch.utils.data.DataLoader(CustomImageDataset(x, y, transform), batch_size=16, shuffle=True)

  return loader

# Fonction pour entrainer le set de données chez le client

def client_update(client_model, optimizer, train_loader, epoch=5):
    """
    This function updates/trains client model on client data
    """
    model.train()
    for e in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = client_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
    return loss.item()

# Donne le model global actuel au client -> partir tous sur la meme base mais pas avec les meme donnée

def client_syn(client_model, global_model):
  '''
  This function synchronizes the client model with global model
  '''
    client_model.load_state_dict(global_model.state_dict())

# Rassemblement de tout les model


def server_aggregate(global_model, client_models,client_lens):
    """
    This function has aggregation method 'wmean'
    wmean takes the weighted mean of the weights of models
    """
    total = sum(client_lens)
    n = len(client_models)
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float()*(n*client_lens[i]/total) for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    for model in client_models:
        model.load_state_dict(global_model.state_dict())


# test du nouveau model global sur le dataset de base ???
"""marche comment en vrai parce que un set global/ de base c'est pas censé exister """


def test(global_model, test_loader):
    """
    This function test the global model on test
    data and returns test loss and test accuracy
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = global_model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)

    return test_loss, acc


#### Initialisation des model et des optimiseur de model

#### global model ##########
global_model =  VGG('VGG19').cuda()

############# client models ###############################
"""Est-il possible d'avoir des model différent entre client/client ou entre global/client"""
client_models = [ VGG('VGG19').cuda() for _ in range(num_selected)]
for model in client_models:
    model.load_state_dict(global_model.state_dict()) ### initial synchronizing with global modle

###### optimizers ################
opt = [optim.SGD(model.parameters(), lr=0.001) for model in client_models]

####### baseline data ############
loader_fixed = baseline_data(baseline_num)

###### Loading the data using the above function ######
train_loader, test_loader = get_data_loaders(classes_pc=classes_pc, nclients= num_clients,
                                                      batch_size=batch_size,verbose=True)
### Initialisation objet de visualisation
losses_train = []
losses_test = []
acc_test = []
losses_retrain = []


# Runnining FL
for r in range(num_rounds):  # Communication round
    # select random clients
    client_idx = np.random.permutation(num_clients)[:num_selected]
    client_lens = [len(train_loader[idx]) for idx in client_idx]

    # client update
    loss = 0
    for i in tqdm(range(num_selected)):
        client_syn(client_models[i], global_model)
        loss += client_update(client_models[i], opt[i], train_loader[client_idx[i]], epochs)
    losses_train.append(loss)

    # server aggregate
    #### retraining on the global server
    loss_retrain = 0
    for i in tqdm(range(num_selected)):
        loss_retrain += client_update(client_models[i], opt[i], loader_fixed, iterations=retrain_epochs)
    losses_retrain.append(loss_retrain)

    ### Aggregating the models
    server_aggregate(global_model, client_models, client_lens)
    test_loss, acc = test(global_model, test_loader)
    losses_test.append(test_loss)
    acc_test.append(acc)
    print('%d-th round' % r)
    print(
        'average train loss %0.3g | test loss %0.3g | test acc: %0.3f' % (loss_retrain / num_selected, test_loss, acc))

