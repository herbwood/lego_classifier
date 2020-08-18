import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class LegoAlexNet(BaseModel):

  def __init__(self, num_classes=10):
    super(LegoAlexNet, self).__init__()                      # (batch size, channel, width, height)
    self.features = nn.Sequential(                       # input : (64, 3, 400, 400)
        nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  #  (64, 64, 99, 99)
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),                  #  (64, 64, 49, 49)
        nn.Conv2d(64, 192, kernel_size=5, padding=2),           #  (64, 192, 49, 49)
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),                  #  (64, 192, 24, 24)
        nn.Conv2d(192, 384, kernel_size=3, padding=1),          #  (64, 384, 24, 24)
        nn.ReLU(inplace=True),
        nn.Conv2d(384, 256, kernel_size=3, padding=1),          #  (64, 256, 24, 24)
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),          #  (64, 256, 24, 24)
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),                  #  (64, 256, 11, 11)
    )
    self.avgpool = nn.AdaptiveAvgPool2d((6, 6))                 #  (64, 256, 6, 6)
    self.classifier = nn.Sequential(
        nn.Dropout(),
        nn.Linear(256 * 6 * 6, 4096),                           #  (64, 4096)
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, 4096),                                  #  (4096, 4096)
        nn.ReLU(inplace=True),
        nn.Linear(4096, num_classes),                           #  (4096, 10)
    )

  def forward(self, x):
    x = self.features(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return F.log_softmax(x, dim=1)