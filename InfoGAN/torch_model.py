import random
import matplotlib.pyplot as plt
import numpy as np
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import os

from tensorflow.examples.tutorials.mnist import input_data


# pytorch == 0.4.0

def NLL_Gaussian(input):
    mean = torch.mean(input, dim=0)
    sigma = torch.std(input, dim=0)
    sigma = sigma * sigma
    loss = torch.mean(0.5 * torch.log(sigma) + torch.mean((input - mean) ** 2 / 2 / sigma, dim=0))
    return loss


def one_hot(labels):
    labels = torch.LongTensor(labels)
    out = torch.zeros([labels.size()[0], 10])
    out.scatter_(1, labels.unsqueeze(dim=1), 1)
    if torch.cuda.is_available():
        return out.cuda()
    return out


# for Discriminator

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        fe1 = list()
        pad = nn.ReplicationPad2d(1)
        lRelu = nn.LeakyReLU(0.2)
        fe1.append(pad)
        fe1.append(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(4, 4), stride=2))
        fe1.append(pad)
        fe1.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=2))
        fe1.append(nn.BatchNorm2d(128))
        fe2 = list()
        fe2.append(nn.Linear(in_features=7 * 7 * 128, out_features=1024))
        fe2.append(nn.BatchNorm1d(1024))
        self.FE1 = nn.Sequential(*fe1)
        self.FE2 = nn.Sequential(*fe2)
        self.D = nn.Linear(in_features=1024, out_features=1)

        q = list()
        q.append(nn.Linear(1024, 128))
        q.append(nn.BatchNorm1d(128))
        q.append(nn.LeakyReLU(0.2))
        # q.append(nn.Linear(128, 12))
        self.Q = nn.Sequential(*q)
        self.Q_cont = nn.Sequential(nn.Linear(128, 2), nn.Sigmoid())
        self.Q_disc = nn.Sequential(nn.Linear(128, 10), nn.Softmax(dim=1))
        # 2 for continuous latent codes, 10 for category.
        # For this task, we use 1 ten-dimensional categorical code, 2 continuous
        # latent codes and 62 noise variables, resulting in a concatenated dimension of 74.

    def forward(self, *input):
        x = input[0]
        # x = torch.reshape(x, (-1, 1, 28, 28))
        x = self.FE1(x)
        x = torch.reshape(x, (-1, 7 * 7 * 128))
        x = self.FE2(x)
        D = self.D(x)
        Q = self.Q(x)
        code_cont = self.Q_cont(Q)
        code_disc = self.Q_disc(Q)
        D = F.sigmoid(D)

        return D, code_disc, code_cont


# for Generator
class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        # self.lRelu = nn.LeakyReLU(0.2)
        self.fc1 = nn.Linear(noise_n + 12, 1024, True)
        # nn.init.xavier_normal_(self.fc1.weight)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 7 * 7 * 128)
        self.bn2 = nn.BatchNorm1d(7 * 7 * 128)
        self.h1 = nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=2, padding=1)  # 7,7 -> 14,14
        self.bn3 = nn.BatchNorm2d(64)
        # nn.init.xavier_normal_(self.h1.weight)
        self.h2 = nn.ConvTranspose2d(64, 1, kernel_size=(4, 4), stride=2, padding=1)  # 14,14 -> 28,28
        nn.init.xavier_normal_(self.h2.weight)
        # self.bn2 = nn.BatchNorm1d(512)
        # self.h2.weight.data.normal_(0, 0.01)
        # self.h3 = nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2)
        # self.h3.weight.data.normal_(0, 0.01)

    def forward(self, *input):
        x = torch.cat(input, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = torch.reshape(x, (-1, 128, 7, 7))
        x = self.h1(x)
        x = F.relu(x)
        x = self.bn3(x)
        x = self.h2(x)
        x = F.sigmoid(x)
        return x


def make_noise(batch_size, noise_n):
    return np.random.normal(size=[batch_size, noise_n])


if __name__ == "__main__":
    torch.set_printoptions(profile="full")
    torch.manual_seed(1234)

    mnist = input_data.read_data_sets("../MNIST_data/")

    Lambda = 1
    learning_rate = 0.0002
    training_epochs = 1000
    batch_size = 64
    noise_n = 62
    flag = 0
    G = Generator().cuda()
    D = Discriminator().cuda()
    opt_G = torch.optim.Adam(
        [{'params': G.parameters()}, {'params': D.Q.parameters()},
         {'params': D.Q_disc.parameters()}, {'params': D.Q_cont.parameters()}],
        lr=learning_rate)
    # opt_G = torch.optim.RMSprop(G.parameters())
    opt_D = torch.optim.Adam(
        [{'params': D.D.parameters()}, {'params': D.FE1.parameters()}, {'params': D.FE2.parameters()}],
        lr=learning_rate)
    total_batch = int(mnist.train.num_examples / batch_size)
    now = datetime.datetime.now()
    now = '%02d_%02d_%02d_%02d' % (now.month, now.day, now.hour, now.minute)
    writer = SummaryWriter("log_dir/{}".format(now))

    for epoch in range(training_epochs):
        for i in range(total_batch):
            G.train()
            D.train()
            X, label = mnist.train.next_batch(batch_size, shuffle=True)
            label = one_hot(label)
            X = torch.Tensor(X).cuda()
            X.requires_grad_()
            X = torch.reshape(X, (-1, 1, 28, 28))

            noise = torch.Tensor(make_noise(batch_size, noise_n)).cuda()
            code = torch.Tensor(make_noise(batch_size, 2)).cuda()
            noise.requires_grad_()
            opt_D.zero_grad()
            D_real, _, _ = D(X)
            D_fake, _, _ = D(G(noise, label, code))
            D_loss = -(torch.mean(torch.log(D_real) + torch.log(1 - D_fake)))
            D_loss.backward(retain_graph=True)
            opt_D.step()

            opt_G.zero_grad()
            D_fake, c_disc, c_cont = D(G(noise, label, code))
            G_loss = -torch.mean(torch.log(D_fake)) - Lambda * (
                    torch.mean(label * torch.log(c_disc)) + NLL_Gaussian(c_cont))
            G_loss.backward(retain_graph=True)
            opt_G.step()

            if i % 10 == 0:
                writer.add_scalar("D_loss", D_loss, i + epoch * total_batch)
                writer.add_scalar("G_loss", G_loss, i + epoch * total_batch)
            if i % 100 == 0:
                print("EPOCH : {}, BATCH: {}\n".format(epoch, i), "D_loss : {}, G_loss : {}".format(D_loss, G_loss))
        test_noise = torch.unsqueeze(noise[batch_size // 2], 0)
        test_code = torch.unsqueeze(code[batch_size // 2], 0)
        writer.add_image("Epoch:{}".format(epoch),
                         torch.reshape(
                             G.eval()(test_noise, one_hot([5]), test_code),
                             (28, 28)))
        os.makedirs('models/{}'.format(now), exist_ok=True)
        torch.save(D, 'models/{}/D_{}.pt'.format(now, epoch))
        torch.save(G, 'models/{}/G_{}.pt'.format(now, epoch))

    print('Learning finished')
