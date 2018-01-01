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

from modules.trainer import Trainer

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
end_epoch = 50
data_root = ''

# Preprocessing
transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
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
trainer = Trainer(optimizer, criterion, model, 100, train_loader, args.use_cuda)
trained_model = trainer.run()
torch.save(trained_model.state_dict(), '../weights/vgg_weight_v1_'+ str(epoch) +'.pth')
