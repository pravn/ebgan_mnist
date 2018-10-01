import argparse
import torch
import os
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.datasets import MNIST
#from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z
        self.res_stride = 1
        self.res_dropout_ratio = 0.0

        '''
        self.fourth = nn.Sequential(
            nn.ConvTranspose2d(self.n_z, 64, 3, 1, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,inplace=True)
            )


        self.third = nn.Sequential(nn.ConvTranspose2d(64, 32, 5, 2, 1, bias=False),
                                   nn.BatchNorm2d(32),
                                   nn.LeakyReLU(0.2, inplace=True)
                                   #nn.ReLU(True)
                                   )

        self.second = nn.Sequential(nn.ConvTranspose2d(32,16,4,2,1,bias=False),
                                    nn.BatchNorm2d(16),
                                    nn.LeakyReLU(0.2,inplace=True)
                                    #nn.ReLU(True)
                                    ) #14x14
            
        self.first = nn.Sequential(nn.ConvTranspose2d(16,1,4,2,1,bias=False),
                                   )

        self.final = nn.Linear(28,28)'''


        self.first = nn.Linear(self.n_z,800)
        self.second = nn.Linear(800,800)
        self.third = nn.Linear(800,784)
        #self.fourth = nn.Linear(800,800)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU(0.2,inplace=1)

    def forward(self, x):
        x = x.view(-1, self.n_z)
        x = self.leaky_relu(self.first(x))
        x = self.leaky_relu(self.second(x))
        x = self.leaky_relu(self.third(x))
        #x = self.leaky_relu(self.fourth(x))
        x = self.tanh(x)
        return x.view(-1,28,28)


class Encoder(nn.Module):
    def __init__(self,args):
        super(Encoder, self).__init__()
        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z
         
        '''
        self.first = nn.Sequential(nn.Conv2d(self.n_channel, 16, 4, 2, 1),
                                    #nn.BatchNorm2d(16),
                                    #nn.LeakyReLU(0.2,inplace=True)
                                    nn.ReLU(True)
                                    )
         self.second = nn.Sequential(nn.Conv2d(16,32,4,2,1),
                                     nn.BatchNorm2d(32),
                                     #nn.LeakyReLU(0.2,inplace=True),
                                     nn.ReLU(True)
                                     )

         self.third = nn.Sequential(nn.Conv2d(32,64,5,2,1),
                                    nn.BatchNorm2d(64),
                                    #nn.LeakyReLU(0.2,inplace=True)
                                    nn.ReLU(True))

         self.fourth = nn.Sequential(nn.Conv2d(64,self.n_z,3,1,0),
                                     nn.BatchNorm2d(self.n_z),
                                     #nn.LeakyReLU()
                                     nn.ReLU(True)
                                     )'''


        self.first = nn.Linear(784,256)
        self.second = nn.Linear(256,self.n_z)
        #self.third = nn.Linear(1024,self.n_z)

        self.relu = nn.ReLU(True)
        self.leaky_relu = nn.LeakyReLU(0.2,inplace=True)

    def forward(self,x):
        x = x.view(-1,784)
        x = self.leaky_relu(self.first(x))
        x = self.leaky_relu(self.second(x))
        #x = self.leaky_relu(self.third(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        
        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z
        
        self.encoder = Encoder(args)
        #self.decoder = Decoder(args)
        self.decoder = nn.Linear(256,784)
        self.relu = nn.ReLU(True)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.encoder(x)
        x = x.squeeze()
        x = self.decoder(x)
        #x = self.tanh(x)
        return  x.view(-1,28,28)


def return_model(args):
    decoder = Decoder(args)
    disc = Discriminator(args)

    decoder = decoder.cuda()
    disc = disc.cuda()

    print('return model - decoder.cuda(), disc.cuda()')

    return decoder, disc
