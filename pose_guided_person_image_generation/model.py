import torch 
import torch.nn as nn

# Convolutional neural network (two convolutional layers)
class Generator1(nn.Module):
    def __init__(self,hidden_num=21):
        super(Generator1, self).__init__()
        #encoder
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(21, hidden_num, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_num, hidden_num, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_num, hidden_num, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
                
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(hidden_num, hidden_num*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_num*2, hidden_num*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_num*2, hidden_num*2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),        
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(hidden_num*2, hidden_num*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_num*4, hidden_num*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_num*4, hidden_num*4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),        
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(hidden_num*4, hidden_num*8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_num*8, hidden_num*8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_num*8, hidden_num*8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),        
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(hidden_num*8, hidden_num*16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_num*16, hidden_num*16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_num*16, hidden_num*16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),        
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(hidden_num*16, hidden_num*32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_num*32, hidden_num*32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_num*32, hidden_num*32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),        
        )
        self.fc = nn.Linear(4*4*hidden_num*32,4*4*hidden_num*32)
        self.up=nn.Upsample(scale_factor=2, mode='nearest')
        self.dlayer6 = nn.Sequential(
            nn.Conv2d(hidden_num*32, hidden_num*16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_num*16, hidden_num*16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_num*16, hidden_num*16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),        
        )
        self.dlayer5 = nn.Sequential(
            nn.Conv2d(hidden_num*32, hidden_num*8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_num*8, hidden_num*8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_num*8, hidden_num*8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),        
        )
        self.dlayer4 = nn.Sequential(
            nn.Conv2d(hidden_num*16, hidden_num*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_num*4, hidden_num*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_num*4, hidden_num*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),        
        )
        self.dlayer3 = nn.Sequential(
            nn.Conv2d(hidden_num*8, hidden_num*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_num*2, hidden_num*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_num*2, hidden_num*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),        
        )
        self.dlayer2 = nn.Sequential(
            nn.Conv2d(hidden_num*4, hidden_num, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_num, hidden_num, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_num, hidden_num, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),        
        )
        self.dlayer1 = nn.Sequential(
            nn.Conv2d(hidden_num*2, hidden_num, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_num, hidden_num, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_num, hidden_num, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),        
        )
        self.dlayer0 = nn.Sequential(
            nn.Conv2d(hidden_num, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),        
        )
        
    def forward(self, x1,x2):
        x=torch.cat((x1,x2),1)
        e_out=[]
        out = self.layer1(x)
        e_out.append(out)
        out = self.layer2(out)
        e_out.append(out)
        out = self.layer3(out)
        e_out.append(out)
        out = self.layer4(out)
        e_out.append(out)
        out = self.layer5(out)
        e_out.append(out)
        out = self.layer6(out)
        out = out.reshape(out.size(0), -1)
        
        out = self.fc(out)
       
        out = self.fc(out)
        
        out=out.view(1,-1,4,4)
        
        out=self.up(out)
        #decoder
        out=self.dlayer6(out)
        
        out=self.up(out)
        out=torch.cat((out,e_out[4]),1)
        out=self.dlayer5(out)
        out=self.up(out)
        out=torch.cat((out,e_out[3]),1)
        out=self.dlayer4(out)
        out=self.up(out)
        out=torch.cat((out,e_out[2]),1)
        out=self.dlayer3(out)
        
        out=self.up(out)
        out=torch.cat((out,e_out[1]),1)
        out=self.dlayer2(out)
        
        out=self.up(out)
        out=torch.cat((out,e_out[0]),1)
        out=self.dlayer1(out)
        
        
        out=self.up(out)
        out=self.dlayer0(out)
        return out
# Convolutional neural network (two convolutional layers)
# no fully connected layer
class Generator2(nn.Module):
    def __init__(self,hidden_num=21):
        super(Generator2, self).__init__()
        #encoder
        self.layer1 = nn.Sequential(
            nn.Conv2d(6, hidden_num, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_num, hidden_num, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_num, hidden_num, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
                
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(hidden_num, hidden_num*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_num*2, hidden_num*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_num*2, hidden_num*2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),        
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(hidden_num*2, hidden_num*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_num*4, hidden_num*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_num*4, hidden_num*4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),        
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(hidden_num*4, hidden_num*8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_num*8, hidden_num*8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_num*8, hidden_num*8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),        
        )
        
        
        
        self.up=nn.Upsample(scale_factor=2, mode='nearest')
        
        self.dlayer4 = nn.Sequential(
            nn.Conv2d(hidden_num*24, hidden_num*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_num*4, hidden_num*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_num*4, hidden_num*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),        
        )
        self.dlayer3 = nn.Sequential(
            nn.Conv2d(hidden_num*12, hidden_num*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_num*2, hidden_num*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_num*2, hidden_num*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),        
        )
        self.dlayer2 = nn.Sequential(
            nn.Conv2d(hidden_num*6, hidden_num, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_num, hidden_num, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_num, hidden_num, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),        
        )
        self.dlayer1 = nn.Sequential(
            nn.Conv2d(hidden_num*3, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),        
        )
        
        
    def forward(self, x1,x2):
        x=torch.cat((x1,x2),1)

        e_out=[]
        out = self.layer1(x)
        e_out.append(out)
        
        out = self.layer2(out)
        e_out.append(out)
        
        out = self.layer3(out)
        e_out.append(out)
        
        out = self.layer4(out)
        e_out.append(out)
        
        noise = 2 * torch.rand_like(out) - 1
        out = torch.cat((out,noise), dim=1)
        out = torch.cat((out,e_out[3]), dim=1)
        
        out=self.dlayer4(out)
        out=self.up(out)
        
        noise = 2 * torch.rand_like(out) - 1
        out = torch.cat((out,noise), dim=1)
        out = torch.cat((out,e_out[2]), dim=1)
        out=self.dlayer3(out)
        out=self.up(out)
        
        noise = 2 * torch.rand_like(out) - 1
        out = torch.cat((out,noise), dim=1)
        out = torch.cat((out,e_out[1]), dim=1)
        out=self.dlayer2(out)
        out=self.up(out)
        
        noise = 2 * torch.rand_like(out) - 1
        out = torch.cat((out,noise), dim=1)
        out = torch.cat((out,e_out[0]), dim=1)
        out=self.dlayer1(out)
        out=self.up(out)
        
        return out
class Discriminator(nn.Module):
    def __init__(self, input_dim=6, dim=64):
        super(Discriminator, self).__init__()

        self.conv1  = nn.Sequential(nn.Conv2d(input_dim, dim, 5, 2, 2),
                                    nn.ReLU())
        self.conv2  = nn.Sequential(nn.Conv2d(dim, dim*2, 5, 2, 2),
                                    nn.BatchNorm2d(dim*2),
                                    nn.ReLU())
        self.conv3  = nn.Sequential(nn.Conv2d(dim*2, dim*4, 5, 2, 2),
                                    nn.BatchNorm2d(dim*4),
                                    nn.ReLU())
        self.conv4  = nn.Sequential(nn.Conv2d(dim*4, dim*8, 5, 2, 2),
                                    nn.BatchNorm2d(dim*8),
                                    nn.ReLU())
        self.conv5  = nn.Sequential(nn.Conv2d(dim*8, dim*8, 5, 2, 2),
                                    nn.BatchNorm2d(dim*8),
                                    nn.ReLU())

        self.linear = nn.Sequential(nn.Linear(8*8*8*dim, 1),nn.Sigmoid())
    
    def forward(self, x1,x2):
        x = torch.cat((x1,x2),1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.shape[0],-1)
        x = self.linear(x)
        return x.view(1,-1)