import torch
from torch import nn
from torch.nn import functional as F

class HARmodel(nn.Module):
    def __init__(self):
        super(HARmodel, self).__init__()
        self.conv1 = nn.Conv1d(12,32,5)
        self.conv2 = nn.Conv1d(32,128, 5)
        self.conv3 = nn.Conv1d(128, 64, 5)
       # self.conv4 = nn.Conv1d(100, 32, 5)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        #self.bn2 = nn.BatchNorm1d(196)
        
        self.drop = nn.Dropout(p=0.3)
        self.maxpool1 = nn.MaxPool1d(1, stride=1)
        x = torch.randn(1, 12, 256)
        self._to_linear = None
        self.convs(x)
    
        self.fc = nn.Linear(self._to_linear, 8)
        #self.fc2 = nn.Linear(2, 4)
        #self.fc3 = nn.Linear(4, 8)
        self.fc4 = nn.Linear(8, 4)
        self.fc5 = nn.Linear(4, classes)
    
    
    def convs(self, x):

        x = (self.drop(self.maxpool1(self.bn1(F.relu(self.conv1(x))))))
        x = ((self.maxpool1(self.bn2(F.relu(self.conv2(x))))))
        x = (self.drop(self.maxpool1(self.bn3(F.relu(self.conv3(x))))))
        #x = ((self.maxpool1((F.relu(self.conv4(x))))))
    
        
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]
        return x
        
    def forward(self, x):

        x = torch.reshape(x, (-1, 12, 256))
        x = (self.drop(self.maxpool1(self.bn1(F.relu(self.conv1(x))))))
        x = ((self.maxpool1(self.bn2(F.relu(self.conv2(x))))))
        x = (self.drop(self.maxpool1(self.bn3(F.relu(self.conv3(x))))))
        #x = ((self.maxpool1((F.relu(self.conv4(x))))))
        
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        x = self.fc5(x)

        return x