import argparse
import torch
import os
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import autograd
#from torchvision.datasets import MNIST
#from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F

torch.manual_seed(123)


def plot_loss(loss_array,name):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(loss_array)
    plt.savefig('loss_'+name)


def weights_init_G(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def weights_init_D(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.002)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def run_trainer(train_loader, netD, netG, args):

    margin = args.margin

    optimizerD = optim.Adam(netD.parameters(), lr=args.lr_G, betas=(0.5,0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr_D,betas=(0.5,0.999))

    noise = torch.FloatTensor(args.batch_size, args.n_z, 1, 1)
    noise = noise.cuda()
    noise = Variable(noise)
    

    netG.apply(weights_init_G)
    netD.apply(weights_init_D)


    criterion = nn.MSELoss()

    for epoch in range(1000):
        G_loss_epoch = 0
        recon_loss_epoch = 0
        D_loss_epoch = 0

        data_iter = iter(train_loader)
        i = 0

        for i, (images, labels) in enumerate(train_loader):
            #free_params(netD)
            #frozen_params(netG)
            netD.zero_grad()
            netG.zero_grad()
            
            #print('minmax', torch.min(images[0]), torch.max(images[0]))

            images = Variable(images)
            images = images.cuda()


            for p in netG.parameters():
                p.requires_grad = False

            for p in netD.parameters():
                p.requires_grad = True

            #train disc
            #train disc with real
            output = netD(images)
            #errD_real = torch.abs(output-images).pow(2).mean()
            errD_real = criterion(output.unsqueeze(1),images)

            #print('errD_real', errD_real)

            errD_real.backward(retain_graph=True)

            #train disc with fake
            noise = noise.data.normal_(0,1)
            fake = netG(noise)
            output = netD(fake) 
            #errD_fake = torch.abs(output-fake).pow(2).mean()
            errD_fake = criterion(output,fake.detach())
            errD_fake = margin - errD_fake
            errD_fake = errD_fake.clamp(min=0)

            #    print('errD_fake', errD_fake)
            

            errD_fake.backward()

            errD = (errD_real + errD_fake)/2.0
            D_loss_epoch += errD.data.cpu().item()
            
            optimizerD.step()

            #train G
            for p in netD.parameters():
                p.requires_grad = False

            for p in netG.parameters():
                p.requires_grad = True

            netG.zero_grad()
            
            noise = noise.data.normal_(0,1)
            fake = netG(noise)
            output = netD(fake)
            errG = torch.abs(output-fake).pow(2).mean()
            errG.backward()
            
            G_loss_epoch += errG.mean().data.cpu().item()
            optimizerG.step()

            fake = fake.unsqueeze(1)


            if  i % 200 == 0 :
                print('saving images for batch', i)
                save_image(fake[0:6].data.cpu().detach(), './fake.png')
                save_image(images[0:6].data.cpu().detach(), './real.png')


        #recon_loss_array.append(recon_loss_epoch)
        #d_loss_array.append(d_loss_epoch)

        #if(epoch%5==0):
        #    print('plotting losses')
        #    plot_loss(recon_loss_array,'recon')
        #    plot_loss(d_loss_array, 'disc')

        if(epoch % 1 == 0):
            print("Epoch, G_loss, D_loss" 
                  ,epoch + 1, G_loss_epoch, D_loss_epoch)

