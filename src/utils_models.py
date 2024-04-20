# -*- coding: utf-8 -*-
# Author: Yaokun Su
# Date  : 1/27/2023
# Machine Learning project

from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
import torch
import torch.nn.functional as F


class Mydata(Dataset):
    def __init__(self, root_dir, data_dir, label_file, transform=None):
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.label_file = label_file
        self.transform = transform
        self.data_path = os.path.join(self.root_dir, self.data_dir)
        self.label_path = os.path.join(self.root_dir, self.label_file)
        self.data_list = os.listdir(self.data_path)
        self.labels = []
        with open(self.label_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                data_list = line.strip(']\n').split(' [')[1].split(', ')
                label = [float(data) for data in data_list]
                self.labels.append(label)

    def __getitem__(self, idx):
        data_name = self.data_list[idx]
        data_item_path = os.path.join(self.data_path, data_name)
        data = torch.load(data_item_path)
        name_idx = int(data_name.strip('.pt'))-1
        data_label = self.labels[name_idx]
        data = data.float()  # change it to float
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        data = data.to(device)
        if self.transform:
            data = self.transform(data)
        return data, torch.tensor(data_label)

    def __len__(self):
        return len(self.data_list)


class Mydata_unlabeled(Dataset):
    def __init__(self, root_dir, data_dir, transform=None):
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.transform = transform
        self.data_path = os.path.join(self.root_dir, self.data_dir)
        self.data_list = os.listdir(self.data_path)

    def __getitem__(self, idx):
        data_name = self.data_list[idx]
        data_item_path = os.path.join(self.data_path, data_name)
        data = torch.load(data_item_path)
        data = data.float()  # change it to float
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        data = data.to(device)
        if self.transform:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.data_list)


class FCAE(nn.Module):
    def __init__(self, latent_dim=30):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4144, 400),
                                     nn.Tanh(),
                                     nn.Linear(400, latent_dim),
                                     nn.Tanh()
                                     )
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 400),
                                     nn.Tanh(),
                                     nn.Linear(400, 4144),
                                     nn.Tanh()
                                     )
        self.isfc = True

    def forward(self, x):
        # x_shape = x.shape  # (64, 4144)
        encoded_x = self.encoder(x)
        decoded_x = self.decoder(encoded_x)
        return encoded_x, decoded_x


class FCVAE(nn.Module):
    def __init__(self, latent_dim=30):
        super().__init__()
        self.fc1 = nn.Linear(4144, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 4144)
        self.isfc = True

    def encoder(self, x):
        h1 = nn.functional.tanh(self.fc1(x))
        mu = self.fc21(h1)
        logvar = self.fc22(h1)
        return mu, logvar  # not Tanh after the fc here

    def reparameterization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decoder(self, z):
        h3 = nn.functional.tanh(self.fc3(z))
        return nn.functional.tanh(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterization(mu, logvar)
        return (mu, logvar), self.decoder(z)


class CNNAE(nn.Module):
    def __init__(self, latent_dim=30):
        super().__init__()
        self.latent_dim = latent_dim
        self.isfc = False

        # encoder
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1) # (70,101) -> (35,51)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # (35,51) -> (18,26)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # (18,26) -> (9,13)
        self.fc1 = nn.Linear(128 * 9 * 13, 200)
        self.fc2 = nn.Linear(200, latent_dim)

        # decoder
        self.fc3 = nn.Linear(latent_dim, 200)
        self.fc4 = nn.Linear(200, 128 * 9 * 13)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1) # (9,13) -> (18,26)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0) # (18,26) -> (35,51)
        self.deconv4 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=(1,0)) # (35,51) -> (70,101)

    def encoder(self, x):
        x = x.nan_to_num()
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.tanh(self.conv3(x))
        x = x.view(-1, 128 * 9 * 13)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x

    def decoder(self, z):
        x = F.tanh(self.fc3(z))
        x = F.tanh(self.fc4(x))
        x = x.view(-1, 128, 9, 13)
        x = F.tanh(self.deconv2(x))
        x = F.tanh(self.deconv3(x))
        x = F.tanh(self.deconv4(x))
        return x

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)


class CNNVAE(nn.Module):
    def __init__(self, latent_dim=30):
        super().__init__()
        self.latent_dim = latent_dim
        self.isfc = False

        # encoder
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1) # (70,101) -> (35,51)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # (35,51) -> (18,26)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # (18,26) -> (9,13)
        self.fc1 = nn.Linear(128 * 9 * 13, 200)
        self.fc21 = nn.Linear(200, latent_dim)
        self.fc22 = nn.Linear(200, latent_dim)

        # decoder
        self.fc3 = nn.Linear(latent_dim, 200)
        self.fc4 = nn.Linear(200, 128 * 9 * 13)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1) # (9,13) -> (18,26)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0) # (18,26) -> (35,51)
        self.deconv4 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=(1,0)) # (35,51) -> (70,101)

    def encoder(self, x):
        x = x.nan_to_num()
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.tanh(self.conv3(x))
        x = x.view(-1, 128 * 9 * 13)
        x = F.tanh(self.fc1(x))
        mu = self.fc21(x)
        logvar = self.fc22(x)
        return mu, logvar

    def reparameterization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decoder(self, z):
        x = F.tanh(self.fc3(z))
        x = F.tanh(self.fc4(x))
        x = x.view(-1, 128, 9, 13)
        x = F.tanh(self.deconv2(x))
        x = F.tanh(self.deconv3(x))
        x = F.tanh(self.deconv4(x))
        return x

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterization(mu, logvar)
        return (mu, logvar), self.decoder(z)


def kld(mu, logvar):  # KL Divergence
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class MaskFlatten(object):
    def __init__(self, mask):
        self.mask = mask  # mask is the places where data is valid

    def __call__(self, sample):
        return torch.masked_select(sample, self.mask)


class MaskUnflatten(object):
    def __init__(self, mask):
        self.mask = mask.clone()  # torch.Size([1, 70, 101])
        self.mask = self.mask.float()

    def __call__(self, sample):
        count = 0
        unflattern = self.mask.clone()
        for i in range(len(unflattern[0])):
            for j in range(len(unflattern[0][0])):
                if unflattern[0][i][j]:
                    unflattern[0][i][j] = sample[count]
                    count += 1
                else:
                    unflattern[0][i][j] = torch.nan
        return unflattern


class CodeToFCs_NNN(nn.Module):
    def __init__(self, latent_dim=30):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(latent_dim, 100),
                                   nn.Tanh(),
                                   nn.Linear(100, 1000),
                                   nn.Tanh(),
                                   nn.Linear(1000, 1000),
                                   nn.Tanh(),
                                   nn.Linear(1000, 100),
                                   nn.Tanh(),
                                   nn.Linear(100, 5)
                                   )

    def forward(self, x):
        return self.model(x)

