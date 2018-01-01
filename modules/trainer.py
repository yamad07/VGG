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
        self.losses = []
        for i in range(self.epoch):
            print("Epoch {0}".format(i))
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
            print(loss.data)
            self.losses.append(loss.data)
            self.save_graph()
        return self.model

    def save_graph(self):
        print(self.losses)
        losses = pd.DataFrame(self.losses)
        epoch = pd.DataFrame(list(range(1, len(self.losses))))
        data = pd.concat([epoch, losses])
        data.columns = ["Epoch", "Loss"]
        plt.plot(data)
        plt.xlabel(u"Epoch")
        plt.ylabel(u"Loss")
        plt.save('reslut.jpg')
