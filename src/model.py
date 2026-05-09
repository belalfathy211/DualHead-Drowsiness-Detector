import torch
from torch import nn
from torch.nn import MaxPool2d
import torch.nn.functional as F
class TwoHeadModel(nn.Module):
    def __init__(self, img_size=112, num_classes=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1) #224*224*32
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1)#224*224*32
        self.relu2 = nn.ReLU()
        self.max1 = MaxPool2d(kernel_size=2)#112*112*32
        self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1) #112*112*64
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1)#112*112*64
        self.relu4 = nn.ReLU()
        self.max2 = MaxPool2d(kernel_size=2)#56*56*64
        self.flat = nn.Flatten()#11111
        self.lin1 = nn.Linear(img_size//4*img_size//4*64, num_classes)
        self.lin2 = nn.Linear(img_size//4*img_size//4*64, num_classes)

    def forward(self, X):
        X = self.conv1(X)
        X = self.relu1(X)
        X = self.conv2(X)
        X = self.relu2(X)
        X = self.max1(X)
        X = self.conv3(X)
        X = self.relu3(X)
        X = self.conv4(X)
        X = self.relu4(X)
        X = self.max2(X)
        XF = self.flat(X)
        X1 = F.sigmoid(self.lin1(XF))
        X2 = F.sigmoid(self.lin2(XF))
        return X1,X2

if __name__ == "__main__":
    model = TwoHeadModel()
    X = torch.ones((1,3,112,112))
    y = model(X)
    print(y[0].shape)
    print(y[1].shape)