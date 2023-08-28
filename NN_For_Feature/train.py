import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from Dataset import *
from torch.utils.data import DataLoader
from model.pointnet2d import *

nepoch = 200

batch_size = 20

lr = 0.2

device = torch.device("cuda")

train_dataset = MyData(dataset_path="/media/yuxuan/SSD/ENV_Fea_Train")
# train_dataset.resave_all()
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4)

print("Train set:{}".format(len(train_dataset)))

classifier = PointNetDenseCls2d(k=7)

optimizer = torch.optim.Adam(
    classifier.parameters(),
    lr=lr,
    betas=(0.9, 0.999),
    eps=1e-08,
    # weight_decay=1e-4
)


scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

classifier.cuda()

num_batch = len(train_dataset) / batch_size


class seg_loss(nn.Module):
    def __init__(self):
        super(seg_loss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, pred, label):
        '''

        :param pred: shape=(B, N, C)
        :param label: shape=(B, N)
        :return:
        '''
        loss = self.loss(pred, label)
        return loss


loss_func = seg_loss().to(device)

log_dir = "./logs"
tensorboard_dir = os.path.join(log_dir, 'tensorboard')
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)
for f in os.listdir(log_dir + "/tensorboard"):
    os.remove(log_dir + "/tensorboard/" + f)

writer = SummaryWriter(tensorboard_dir)

min_loss = 100
for epoch in range(nepoch):
    loss_epoch = 100
    for i, data in enumerate(train_loader, 0):
        target = data[0].to(torch.float32)
        points = data[1].to(torch.float32)
        points = points.transpose(2, 1)
        target = target.type(torch.LongTensor)
        points, target = points.to(device), target.to(device)
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points)
        # print("Pred:{}".format(np.shape(pred)))
        # print("Label:{}".format(np.shape(target)))
        loss = loss_func(pred, target)
        loss.backward()
        optimizer.step()
        print('[%d: %d/%d] train loss: %f' % (epoch, i, num_batch, loss.item()))
        if loss < min_loss:
            min_loss = loss
            torch.save(classifier.state_dict(), "model.pth")
    scheduler.step()
    loss_epoch = min_loss
    writer.add_scalar("loss", loss_epoch, epoch)
