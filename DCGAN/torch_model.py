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

# for Discriminator


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.pad = nn.ReplicationPad2d(1)
        self.h1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3))
        # self.h1.weight.data.normal_(0, 0.01)
        self.h2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        # self.h2.weight.data.normal_(0, 0.01)
        self.h3 = nn.Linear(in_features=28 * 28 * 64, out_features=625)
        # self.h3.weight.data.normal_(0, 0.01)
        self.h4 = nn.Linear(in_features=625, out_features=1)
        # self.h4.weight.data.normal_(0, 0.01)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, *input):
        x = input[0]
        # x = torch.reshape(x, (-1, 1, 28, 28))
        x = self.pad(x)
        x = self.h1.forward(x)
        x = F.relu(x)
        x = self.pad(x)
        x = self.h2.forward(x)
        x = F.relu(x)
        x = torch.reshape(x, (-1, 28 * 28 * 64))
        x = self.h3.forward(x)
        x = F.relu(x)
        x = self.h4.forward(x)
        x = self.dropout.forward(x)
        x = F.sigmoid(x)

        return x


# for Generator
class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.projection = nn.Linear(noise_n, 7*7*16, True)
        nn.init.xavier_normal_(self.projection.weight)
        self.h1 = nn.ConvTranspose2d(16, 4, kernel_size=2, stride=2)  # 7,7 -> 14,14
        nn.init.xavier_normal_(self.h1.weight)
        self.bn1 = nn.BatchNorm2d(4)
        self.h2 = nn.ConvTranspose2d(4, 1, kernel_size=2, stride=2)   # 14,14 -> 28,28
        nn.init.xavier_normal_(self.h2.weight)
        # self.bn2 = nn.BatchNorm1d(512)
        # self.h2.weight.data.normal_(0, 0.01)
        # self.h3 = nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2)
        # self.h3.weight.data.normal_(0, 0.01)

    def forward(self, *input):
        x = input[0]
        x = self.projection(x)
        x = torch.reshape(x, (-1, 16, 7, 7))
        x = self.h1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.h2(x)
        # x = F.relu(x)
        # x = self.bn2(x)
        # x = self.h3(x)
        x = F.sigmoid(x)
        return x


def make_noise(batch_size, noise_n):
    return np.random.normal(size=[batch_size, noise_n])


if __name__ == "__main__":
    torch.set_printoptions(profile="full")
    torch.manual_seed(1234)

    mnist = input_data.read_data_sets("../MNIST_data/")

    learning_rate = 0.0002
    training_epochs = 1000
    batch_size = 64
    noise_n = 100
    flag = 0

    G = Generator().cuda()
    D = Discriminator().cuda()
    opt_G = torch.optim.Adam(G.parameters(), lr=learning_rate)
    # opt_G = torch.optim.RMSprop(G.parameters())
    opt_D = torch.optim.Adam(D.parameters(), lr=learning_rate)
    total_batch = int(mnist.train.num_examples / batch_size)
    now = datetime.datetime.now()
    now = '%02d_%02d_%02d_%02d' % (now.month, now.day, now.hour, now.minute)
    writer = SummaryWriter("log_dir/{}".format(now))

    for epoch in range(training_epochs):
        for i in range(total_batch):
            G.train()
            D.train()
            X, _ = mnist.train.next_batch(batch_size)
            X = torch.Tensor(X).cuda()
            X.requires_grad_()
            X = torch.reshape(X, (-1, 1, 28, 28))

            noise = torch.Tensor(make_noise(batch_size, noise_n)).cuda()
            noise.requires_grad_()

            opt_D.zero_grad()
            D_real = D(X)
            D_fake = D(G(noise))
            D_loss = -(torch.mean(torch.log(D_real) + torch.log(1 - D_fake)))
            D_loss.backward(retain_graph=True)
            opt_D.step()

            opt_G.zero_grad()
            D_fake = D(G(noise))
            G_loss = -torch.mean(torch.log(D_fake))
            G_loss.backward(retain_graph=True)
            opt_G.step()

            if i % 10 == 0:
                writer.add_scalar("D_loss", D_loss, i + epoch * total_batch)
                writer.add_scalar("G_loss", G_loss, i + epoch * total_batch)
            if i % 100 == 0:
                print("EPOCH : {}, BATCH: {}\n".format(epoch, i), "D_loss : {}, G_loss : {}".format(D_loss, G_loss))
        writer.add_image("Epoch:{}".format(epoch),
                         torch.reshape(G.eval()(torch.unsqueeze(noise[batch_size // 2], 0)), (28, 28)))
        os.makedirs('models/{}'.format(now), exist_ok=True)
        torch.save(D, 'models/{}/D_{}.pt'.format(now, epoch))
        torch.save(G, 'models/{}/G_{}.pt'.format(now, epoch))

    print('Learning finished')
