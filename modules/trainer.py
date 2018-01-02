import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd


class Trainer:

    def __init__(self, optimizer, criterion, model, epoch, loader, use_cuda):
        self.optimizer = optimizer
        self.criterion = criterion
        self.model = model
        self.epoch = epoch
        self.loader = loader
        self.use_cuda = use_cuda

    def run(self):
        self.average_losses = []
        losses = torch.Tensor([0])
        for e in range(1, self.epoch + 1):
            for i, (inputs, labels) in enumerate(self.loader):
                if self.use_cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                inputs = Variable(inputs)
                labels = Variable(labels)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                print('Epoch: {0}/{1} [{2}/{3}] Loss: {4}'.format(e, self.epoch, i, len(self.loader), loss.data[0]))
                losses.add(loss)
            average_loss = losses.div(len(self.loader))
            self.average_losses.append(average_loss.data[0])
        self.save_graph()
        return self.model

    def save_graph(self):
        print(self.losses)
        losses = pd.DataFrame(self.losses)
        plt.plot(losses)
        plt.xlabel(u"Epoch")
        plt.ylabel(u"Loss")
        plt.savefig('reslut.jpg')

