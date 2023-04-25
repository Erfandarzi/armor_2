from torch import nn
from efficientnet_pytorch import EfficientNet


class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR, self).__init__()
        self.conv2d_1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(4096, 512)
        self.fc_2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x


class CNN_MNIST(nn.Module):
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        self.conv2d_1 = nn.Conv2d(1, 4, kernel_size=5, stride=1)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = nn.Conv2d(4, 10, kernel_size=5, stride=1)
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(160, 100)
        self.fc_2 = nn.Linear(100, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x


class CNN_MRI(nn.Module):
    def __init__(self):
        super(CNN_MRI,self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 100, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(100, 150, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 

            nn.Conv2d(150, 200, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(200, 200, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 

            nn.Conv2d(200, 250, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(250, 250, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 

            nn.Flatten(), 
            nn.Linear(36000, 64),  
            nn.ReLU(),            
            nn.Linear(64, 32),  
            nn.ReLU(),            
            nn.Linear(32, 16),           
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(8, 2))
        
    def forward(self, xb):
        return self.network(xb)

# class CNN_netNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 =             nn.Conv2d(3, 100, kernel_size=3, padding=1),
# sel
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

class CNN_Pathologym(nn.Module):
    def __init__(self):
            super(CNN_Pathologym,self).__init__()
            self.conv1=nn.Conv2d(3, 100, kernel_size=3, padding=1)
            self.relu1=nn.ReLU()
            self.conv2=nn.Conv2d(100, 150, kernel_size=3, stride=1, padding=1)
            self.relu2=nn.ReLU()
            self.pooling1=nn.MaxPool2d(2, 2)

            self.conv3=nn.Conv2d(150, 200, kernel_size=3, stride=1, padding=1)
            self.relu3=nn.ReLU()
            self.conv4= nn.Conv2d(200, 200, kernel_size=3, stride=1, padding=1)
            self.relu4=nn.ReLU()
            self.pooling2=nn.MaxPool2d(2, 2)

            self.conv5=nn.Conv2d(200, 250, kernel_size=3, stride=1, padding=1)
            self.relu5=nn.ReLU()
            self.conv6=nn.Conv2d(250, 250, kernel_size=3, stride=1, padding=1)
            self.relu6=nn.ReLU()
            self.pooling3=nn.MaxPool2d(2, 2)

            self.flatten1=nn.Flatten()
            self.fc1=nn.Linear(36000, 64)
            self.relu7=nn.ReLU()          
            self.fc2=nn.Linear(64, 32)  
            self.relu8=nn.ReLU()            
            self.fc3=nn.Linear(32, 16)           
            self.relu9=nn.ReLU()
            self.fc4=nn.Linear(16, 8)
            self.relu10=nn.ReLU()
            self.dropout1=nn.Dropout(0.25)
            self.fc5=nn.Linear(8, 2)
        
    def forward(self, xb):
        x=self.conv1(xb)
        x=self.relu1(x)
     
        x=self.conv2(x)
        x=  self.relu2(x)
        x=self.pooling1(x)

        x=     self.conv3(x)
        x=   self.relu3(x)
        x=  self.conv4(x)
        x= self.relu4(x)
        x=       self.pooling2(x)

        x=     self.conv5(x)
        x=   self.relu5(x)
        x= self.conv6(x)
        x=   self.relu6(x)
        x=  self.pooling3(x)

        x=  self.flatten1(x)
        x= self.fc1(x)
        x=    self.relu7(x)         
        x=   self.fc2  (x)
        x=   self.relu8   (x)         
        x=  self.fc3         (x)  
        x= self.relu9(x)
        x=  self.fc4(x)
        x=  self.relu10(x)
        x= self.dropout1(x)
        x= self.fc5(x)

        return x

class CNN_Pathology(nn.Module):
    def __init__(self):
        super(CNN_Pathology,self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 100, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(100, 150, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 

            nn.Conv2d(150, 200, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(200, 200, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 

            nn.Conv2d(200, 250, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(250, 250, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 

            nn.Flatten(), 
            nn.Linear(36000, 64),  
            nn.ReLU(),            
            nn.Linear(64, 32),  
            nn.ReLU(),            
            nn.Linear(32, 16),           
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(8, 2))
        
    def forward(self, xb):
        return self.network(xb)

class CNN_Tumors(nn.Module):
    def __init__(self):
        super(CNN_Tumors,self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 100, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(100, 150, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 

            nn.Conv2d(150, 200, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(200, 200, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 

            nn.Conv2d(200, 250, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(250, 250, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 

            nn.Flatten(), 
            nn.Linear(12250, 64),  
            nn.ReLU(),            
            nn.Linear(64, 32),  
            nn.ReLU(),            
            nn.Linear(32, 16),           
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(8,4))
        
    def forward(self, xb):
        return self.network(xb)

# class CNN_MRI(nn.Module):
#     def __init__(self):
#         super(CNN_MRI, self).__init__()
#         self.resnet =  EfficientNet.from_pretrained('efficientnet-b0')
#         self.l1 = nn.Linear(1000 , 256)
#         self.dropout = nn.Dropout(0.75)
#         self.l2 = nn.Linear(256,2)
#         self.relu = nn.ReLU()

#     def forward(self, input):
#         x = self.resnet(input)
#         x = x.view(x.size(0),-1)
#         x = self.dropout(self.relu(self.l1(x)))
#         x = self.l2(x)
#         return x


class CNN_Chest(nn.Module):
    def __init__(self):
        super(CNN_Chest,self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 100, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(100, 150, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 

            nn.Conv2d(150, 200, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(200, 200, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 

            nn.Conv2d(200, 250, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(250, 250, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 

            nn.Flatten(), 
            nn.Linear(16000, 64),  
            nn.ReLU(),            
            nn.Linear(64, 32),  
            nn.ReLU(),            
            nn.Linear(32, 16),           
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(8, 2))
        
    def forward(self, xb):
        return self.network(xb)