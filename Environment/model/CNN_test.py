import torch
import torch.nn as nn
import torch.nn.functional as F


#


#
#
# class classifier(nn.Module):
#     def __init__(self, prob=0.5):
#         super(classifier, self).__init__()
#         self.fc1 = nn.Linear(128, 64)
#         self.bn1_fc = nn.BatchNorm1d(64)
#         self.fc2 = nn.Linear(64, 32)
#         self.bn2_fc = nn.BatchNorm1d(32)
#         self.fc3 = nn.Linear(32, 5)
#         self.bn_fc3 = nn.BatchNorm1d(5)
#         self.prob = prob
#
#     def set_lambda(self, lambd):
#         self.lambd = lambd
#
#     def forward(self, x, reverse=False):
#         if reverse:
#             x = grad_reverse(x, self.lambd)
#         x = F.relu(self.bn1_fc(self.fc1(x)))
#         x = F.relu(self.bn2_fc(self.fc2(x)))
#         x = F.dropout(x, training=self.training)
#         x = self.fc3(x)
#         return x


# class features(nn.Module):
#     def __init__(self):
#         super(features, self).__init__()
#         self.features = nn.Sequential(
#             nn.BatchNorm2d(3),
#             nn.Conv2d(3, 32, kernel_size=(5,5), stride=(1,1),padding = (2,2)),
#             nn.BatchNorm2d(32),
#             nn.ReLU(True),
#             nn.MaxPool2d(stride=2, kernel_size=5),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1,padding = (1,1)),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#             nn.MaxPool2d(stride=1, kernel_size=3),
#             nn.Conv2d(64, 128, kernel_size=3, stride=1,padding = (1,1)),
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1,padding = (1,1)),
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#             Reshape(),
#             nn.Linear(270848, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(True),
#             nn.Dropout())
#
#     def forward(self, x):
#         x = self.features(x)
#         return x
#
#
#
# class classifier(nn.Module):
#     def __init__(self, prob=0.5):
#         super(classifier, self).__init__()
#         self.classifier = nn.Sequential(
#             nn.Linear(128, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(True),
#             nn.Linear(64, 32),
#             nn.BatchNorm1d(32),
#             nn.ReLU(True),
#             nn.Linear(32, 5))
#
#
#     def forward(self, x, reverse=False):
#         x = self.classifier(x)
#         return x
class Reshape(nn.Module):
    def __init__(self):
        super(Reshape, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), 110976)
        return x


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.model = nn.Sequential(
            # nn.BatchNorm2d(1),
            nn.Conv2d(1, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(stride=2, kernel_size=5),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(stride=2, kernel_size=5),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(stride=1, kernel_size=3),
            Reshape(),
            nn.Linear(110976, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 5),
        )

    def forward(self, x):
        x = self.model(x)
        return x

# class Reshape(nn.Module):
#     def __init__(self):
#         super(Reshape, self).__init__()
#
#     def forward(self, x):
#         x = x.view(x.size(0), 102400)
#         return x

# class extractor(nn.Module):
#     def __init__(self):
#         super(extractor, self).__init__()
#         self.extractor = nn.Sequential(
#             nn.BatchNorm2d(3),
#             nn.Conv2d(3, 64, kernel_size=(5,5), stride=(1,1),padding = (2,2)),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#             nn.MaxPool2d(stride=2, kernel_size=5),
#             nn.Conv2d(64, 128, kernel_size=3, stride=1,padding = (1,1)),
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#             nn.MaxPool2d(stride=2, kernel_size=5),
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=(1, 1)),
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
#             nn.MaxPool2d(stride=1, kernel_size=3),
#             Reshape(),
#             nn.Linear(102400, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(True),
#             nn.Dropout()
#         )
#
#     def forward(self, x):
#         x = self.extractor(x)
#         return x
#
#
# class classifier(nn.Module):
#     def __init__(self, prob=0.5):
#         super(classifier, self).__init__()
#         self.classifier = nn.Sequential(
#             nn.Linear(256, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(True),
#             nn.Linear(64, 32),
#             nn.BatchNorm1d(32),
#             nn.ReLU(True),
#             nn.Linear(32, 5))
#
#
#     def forward(self, x, reverse=False):
#         x = self.classifier(x)
#         return x
