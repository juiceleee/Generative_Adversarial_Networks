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
    loss = torch.mean(0.5 * torch.log(sigma) + torch.mean((input - mean) ** 2.0 / 2.0 / sigma, dim=0))
    return loss


def one_hot(labels):
    labels = torch.LongTensor(labels)
    out = torch.zeros([labels.size()[0], 10])
    out.scatter_(1, labels.unsqueeze(dim=1), 1)
    if torch.cuda.is_available():
        return out.cuda()
    return out


# for Discriminator

class Discriminator_pre(nn.Module):

    def __init__(self):
        super(Discriminator_pre, self).__init__()
        self.pad = nn.ReplicationPad2d(1)
        self.h1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3))
        # self.h1.weight.data.normal_(0, 0.01)
        self.h2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        # self.h2.weight.data.normal_(0, 0.01)
        self.h3 = nn.Linear(in_features=28 * 28 * 64, out_features=625)
        # self.h3.weight.data.normal_(0, 0.01)
        # self.h4 = nn.Linear(in_features=625, out_features=1)
        # self.h4.weight.data.normal_(0, 0.01)
        self.dropout = nn.Dropout(p=0.3)

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
        x = self.dropout(x)
        x = self.h3.forward(x)
        x = F.relu(x)
        # x = self.h4.forward(x)
        # x = self.dropout.forward(x)
        # x = F.sigmoid(x)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.h4 = nn.Linear(in_features=625, out_features=1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, *input):
        x = input[0]
        x = self.h4.forward(x)
        x = self.dropout.forward(x)
        x = F.sigmoid(x)

        return x


class Q_net(nn.Module):
    def __init__(self):
        super(Q_net, self).__init__()
        self.h4 = nn.Linear(in_features=625, out_features=10)
        self.h5 = nn.Linear(in_features=625, out_features=2)

    def forward(self, *input):
        x = input[0]
        disc = self.h4(x)
        cont = self.h5(x)

        return F.softmax(disc, 1), F.sigmoid(cont)


# for Generator
class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.projection = nn.Linear(noise_n + 12, 7 * 7 * 16, True)
        nn.init.xavier_normal_(self.projection.weight)
        self.h1 = nn.ConvTranspose2d(16, 4, kernel_size=2, stride=2)  # 7,7 -> 14,14
        nn.init.xavier_normal_(self.h1.weight)
        self.bn1 = nn.BatchNorm2d(4)
        self.h2 = nn.ConvTranspose2d(4, 1, kernel_size=2, stride=2)  # 14,14 -> 28,28
        nn.init.xavier_normal_(self.h2.weight)
        # self.bn2 = nn.BatchNorm1d(512)
        # self.h2.weight.data.normal_(0, 0.01)
        # self.h3 = nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2)
        # self.h3.weight.data.normal_(0, 0.01)

    def forward(self, *input):
        x = torch.cat(input, 1)
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
    torch.manual_seed(123)
    np.random.seed(123)

    mnist = input_data.read_data_sets("../MNIST_data/")

    Lambda = 3
    learning_rate = 0.0002
    training_epochs = 1000
    batch_size = 256
    noise_n = 62
    alpha = 1e-05
    G = Generator().cuda()
    D_front = Discriminator_pre().cuda()
    D = Discriminator().cuda()
    Q = Q_net().cuda()
    # opt_G = torch.optim.Adam(
    #     [{'params': G.parameters()}, {'params': Q.parameters()}],
    #     lr=0.0005)
    opt_Q = torch.optim.Adam(Q.parameters(), lr=0.0005)
    opt_G = torch.optim.Adam(G.parameters(), lr=0.0005)
    opt_D = torch.optim.Adam(
        [{'params': D_front.parameters()}, {'params': D.parameters()}],
        lr=0.0002)
    # opt_D = torch.optim.Adam(D.parameters(), lr=0.0002)
    total_batch = int(mnist.train.num_examples / batch_size)
    now = datetime.datetime.now()
    now = '%02d_%02d_%02d_%02d' % (now.month, now.day, now.hour, now.minute)
    writer = SummaryWriter("log_dir/{}".format(now))

    for epoch in range(training_epochs):
        for i in range(total_batch):
            G.train()
            D.train()
            Q.train()
            D_front.train()
            X, label = mnist.train.next_batch(batch_size, shuffle=True)
            label = one_hot(label)
            label.requires_grad_()
            X = torch.Tensor(X).cuda()
            X.requires_grad_()
            X = torch.reshape(X, (-1, 1, 28, 28))

            noise = torch.Tensor(make_noise(batch_size, noise_n)).cuda()
            code = torch.Tensor(make_noise(batch_size, 2)).cuda()
            code.requires_grad_()
            noise.requires_grad_()

            opt_D.zero_grad()
            # D_real, _, _ = D(X)
            D_real = D(D_front(X))
            pre_fake = D_front(G(noise, label, code))
            # D_fake, _, _ = D((G(noise, label, code)))
            D_fake = D(pre_fake)
            D_loss = -(torch.mean(torch.log(D_real+alpha) + torch.log(1 - D_fake+alpha)))
            D_loss.backward(retain_graph=True)
            opt_D.step()

            opt_Q.zero_grad()
            Q_loss = -torch.mean(label * torch.log(Q(D_front(X))))
            Q_loss.backward(retain_graph=True)
            opt_Q.step()

            opt_G.zero_grad()
            pre_fake = D_front(G(noise, label, code))
            # D_fake, c_disc, c_cont = D(G(noise, label, code))
            D_fake = D(pre_fake)
            c_disc, c_cont = Q(pre_fake)
            real_c_disc, real_c_cont = Q(D_front(X))
            # class_loss = torch.mean(label * torch.log(c_disc+alpha)) + torch.mean(label * torch.log(real_c_disc+alpha))
            # cont_loss = NLL_Gaussian(c_cont) + NLL_Gaussian(real_c_cont)
            class_loss = torch.mean(label * torch.log(c_disc+alpha))
            cont_loss = NLL_Gaussian(c_cont)
            GAN_loss = torch.mean(torch.log(D_fake))
            G_loss = - GAN_loss - Lambda * (2.0*class_loss + cont_loss)
            G_loss.backward(retain_graph=True)
            opt_G.step()
            # for name, param in G.named_parameters():
            #     writer.add_histogram(name, param, i + epoch * total_batch)
            if i % 10 == 0:
                writer.add_scalar("D_loss", D_loss, i + epoch * total_batch)
                writer.add_scalar("Q_loss", Q_loss, i + epoch * total_batch)
                writer.add_scalar("G_loss", G_loss, i + epoch * total_batch)
                writer.add_scalar("class_loss", class_loss, i + epoch * total_batch)
                writer.add_scalar("cont_loss", cont_loss, i + epoch * total_batch)
                writer.add_scalar("GAN_loss", GAN_loss, i + epoch * total_batch)
            if i % 100 == 0:
                print("EPOCH : {}, BATCH: {}\n".format(epoch, i), "D_loss : {}, G_loss : {}".format(D_loss, G_loss))
        test_noise = torch.unsqueeze(noise[batch_size // 2], 0)
        test_code = torch.unsqueeze(code[batch_size // 2], 0)
        for i in range(10):
            writer.add_image("Epoch_{}".format(epoch),
                         torch.reshape(
                             G.eval()(test_noise, one_hot([i]), test_code),
                             (28, 28)), i)
        print("Epoch : {}".format(epoch))
        classification, _ = Q(D_front(G(test_noise, one_hot([5]), test_code)))
        print("Classification : {}".format(classification))
        # print(F.mse_loss(torch.argmax(D(X)[1], dim=1), torch.argmax(label, dim=1)))
        os.makedirs('models/{}'.format(now), exist_ok=True)
        torch.save(D, 'models/{}/D_{}.pt'.format(now, epoch))
        torch.save(G, 'models/{}/G_{}.pt'.format(now, epoch))

    print('Learning finished')
