from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    '''
    Encoder part of the classification network
    '''
    def __init__(self, in_channels=1, dropout=0.5): #, h=256):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 20, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(20)
        self.bn2 = nn.BatchNorm2d(50)
        # self.conv3 = nn.Conv2d(16, 120, kernel_size=4, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        # self.dropout1 = nn.Dropout2d(dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(1250, 500)

        for m in self.modules():
            # Intializing the weights with Kaiming weights as activation function is ReLU
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        '''
        Forward function of the encoder
        '''
        # x: (batch_size,3,32,32)

        bs = x.size(0)
        x = self.relu(self.bn1(self.conv1(x))) # (bs, 20, 28, 28)
        x = self.pool(x) # (bs, 20, 14,14)
        x = self.relu(self.bn2(self.conv2(x))) # (bs, 50, 10, 10)
        x = self.pool(x) # (bs, 50, 5, 5)
        x = x.view(bs, -1) # (bs, 1250)
        x = self.dropout(x)
        x = self.fc(x) # (bs, 500)

        return x


class Classifier(nn.Module):
    '''
    Classifier part of thr classification network
    '''
    def __init__(self, n_classes, dropout=0.5):
        super(Classifier, self).__init__()
        self.l1 = nn.Linear(500, n_classes)

        for m in self.modules():
            # Intializing the weights with Kaiming weights as activation function is ReLU
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        '''
        Forward function of the network
        '''
        # x: (bs, 500 )

        x = self.l1(x) # (bs, 10)
        return x


class CNN(nn.Module):
    '''
    Classification network that has two parts:
        - Encoder
        - Classifier
    '''
    def __init__(self, in_channels=1, n_classes=10, target=False):
        super(CNN, self).__init__()
        self.encoder = Encoder(in_channels=in_channels) # Encoder
        self.classifier = Classifier(n_classes) # Classifier

        # If target is true, classifier part of the network is not trained
        if target:
            for param in self.classifier.parameters():
                param.requires_grad = False

    def forward(self, x):
        '''
        Forward function of the Classification Network
        '''
        
        # x: (batch_size,3,32,32)
        x = self.encoder(x) # (bs, 500)
        x = self.classifier(x) # (bs, 10)
        return x


class Discriminator(nn.Module):
    '''
    Discriminator Network
    '''
    def __init__(self, h=500, args=None):
        super(Discriminator, self).__init__()
        self.l1 = nn.Linear(500, h)
        self.l2 = nn.Linear(h, h)
        self.l3 = nn.Linear(h, 2)
        self.slope = args.slope # slope of leaky ReLU function

        for m in self.modules():
            # Intializing the weights with Kaiming weights as activation function is ReLU
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        '''
        Forward function of Discriminator Network
        '''
        x = F.leaky_relu(self.l1(x), self.slope)
        x = F.leaky_relu(self.l2(x), self.slope)
        x = self.l3(x)
        return x
