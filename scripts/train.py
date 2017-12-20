import sys
sys.path.append('../')
import torch

import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torchvision import models
import argparse

# Parser
parser = argparse.ArgumentParser(description="VGG model coding by yamad07")
parser.add_argument("--use_cuda", default=False, help="gpu or cpu")
args = parser.parse_args()

# Hyper Parameters
num_classes = 3
lr = 0.001
momentum = 0.9
batch_size = 5
start_epoch = 1
end_epoch = 20
data_root = ''

# Preprocessing
transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)
datasets = dset.ImageFolder('../images/', transform=transforms)
train_loader = torch.utils.data.DataLoader(datasets, batch_size=batch_size, shuffle=True)

# Model Setting
model = models.vgg19(pretrained=True)
model.add_module("classefier", nn.Linear(1000, num_classes))
if args.use_cuda:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

def train(model, criterion, optimizer, train_loader, epoch, use_cuda):
    print('Epoch %d' % epoch)
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        if args.use_cuda:
            images = images.cuda()
            labels = labels.cuda()

        images = Variable(images)
        labels = Variable(labels)
        optimizer.zero_grad()

        pred = model(images)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()
        print(loss.data[0])

for epoch in range(start_epoch, end_epoch + 1):
    train(model, criterion, optimizer, train_loader, epoch, args.use_cuda)
    torch.save(model.state_dict(), './weights/vgg_weight_v1.pth')
