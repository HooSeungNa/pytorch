import torch
import torch.nn as nn

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, kernel_size=4, stride=2,
                            padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.ReLU())
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

#up phase convtranspose2d
class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
                 nn.InstanceNorm2d(out_size),
                 nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class Generator(nn.Module):
    def __init__(self,in_channel=3):
        super(Generator,self).__init__()
        self.down1=UNetDown(in_channel,64)
        self.down2=UNetDown(64,128)
        self.down3=UNetDown(128,256)
        self.down4=UNetDown(256,512)
        self.down5=UNetDown(512,1024)
        
        self.up1=UNetUp(1024,512)
#         self.up1_cnn=UnetUpConv(1024,512)
        
        self.up2=UNetUp(1024,256)
        
        self.up3=UNetUp(512,128)
        
        self.up4=UNetUp(256,64)
        
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, 3, 4, padding=1),
            nn.Tanh()
        )
        
    def forward(self,x):
        o1=self.down1(x)
#         print("o1.shape : ",o1.shape)

        o2=self.down2(o1)
#         o2_max=self.max_pool(o2)
#         print("o2.shape : ",o2.shape)
        
        o3=self.down3(o2)
#         o3_max=self.max_pool(o3)
#         print("o3.shape : ",o3.shape)
        
        o4=self.down4(o3)
#         o4_max=self.max_pool(o4)
#         print("o4.shape : ",o4.shape)
        
        o5=self.down5(o4)
#         print("o5.shape : ",o5.shape)
        
        u1=self.up1(o5) # 128 특징 가지고 있다.
        u1_concat=torch.cat((o4,u1),1)
#         u1_conv=self.up1_cnn(u1_concat)
#         print("u1.shape : ",u1.shape,"u1_concat.shape : ",u1_concat.shape)
        u2=self.up2(u1_concat)
        u2_concat=torch.cat((o3,u2),1)
#         u2_conv=self.up2_cnn(u2_concat)
#         print("u2.shape : ",u2.shape,"u2_concat.shape : ",u2_concat.shape)
        u3=self.up3(u2_concat)
        u3_concat=torch.cat((o2,u3),1)
#         u3_conv=self.up3_cnn(u3_concat)
#         print("u3.shape : ",u3.shape,"u3_concat.shape : ",u3_concat.shape)
        u4=self.up4(u3_concat)
        u4_concat=torch.cat((o1,u4),1)
#         u4_conv=self.up4_cnn(u4_concat)
#         print("u4.shape : ",u4.shape,"u4_concat.shape : ",u4_concat.shape)
        output=self.final(u4_concat)
        

        return output 
class Discriminator(nn.Module):
    def __init__(self,in_size=6,out_size=64):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(out_size),
                                    nn.Conv2d(out_size, out_size*2,
                                              4,2,1, bias=False),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(out_size*2),
                                    
                                    nn.Conv2d(out_size*2, out_size*4,
                                              4,2,1, bias=False),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(out_size*4),
                                    nn.Conv2d(out_size*4, out_size*8,
                                              4,2,1, bias=False),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(out_size*8),
                                    
                                    nn.Conv2d(out_size*8, 1,
                                              4,2,1, bias=False)
                                    )

    def forward(self, x1,x2):
#         print(x1.shape,x2.shape)
        x_cat=torch.cat((x1,x2),1)
#         print("x1.shape : ",x1.shape,"x2.shape : ",x2.shape,"cat.shape : ",x_cat.shape)
        out = self.layer1(x_cat)
        return out