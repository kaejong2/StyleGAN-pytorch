import os, sys
import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import itertools

from utils import *
from model import Generator, Discriminator
from dataloader import data_loader


class cycleGAN():
    def __init__(self, args):
        self.args = args
        
        # Define the Network
        self.netG_A2B = Generator(self.args.input_nc, self.args.output_nc).to(device= self.args.device)
        self.netG_B2A = Generator(self.args.output_nc, self.args.input_nc).to(device= self.args.device)
        self.netD_A = Discriminator(self.args.input_nc).to(device=self.args.device)
        self.netD_B = Discriminator(self.args.output_nc).to(device=self.args.device)        
        init_weight(self.netD_B)
        init_weight(self.netD_A)
        init_weight(self.netG_A2B)        
        init_weight(self.netG_B2A)

        # Define Loss function
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()
        
        # Optimizers
        self.optimizerG = torch.optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()), lr=self.args.lr, betas=(self.args.b1, self.args.b2))
        self.optimizerD = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=self.args.lr, betas=(self.args.b1, self.args.b2))

        # Learning rate scheduler
        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizerG, lr_lambda=LambdaLR(self.args.num_epochs, self.args.epoch, self.args.decay_epoch).step)
        self.lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(self.optimizerD, lr_lambda=LambdaLR(self.args.num_epochs, self.args.epoch, self.args.decay_epoch).step)
        
        #dataset
        self.dataloader = data_loader(self.args)


    def run(self, ckpt_path=None, load_ckpt=None, result_path=None):
        for epoch in range(self.args.num_epochs):
            self.netG_A2B.train()
            self.netG_B2A.train()
            self.netD_A.train()
            self.netD_B.train()
            
            loss_G_A2B_train = []
            loss_G_B2A_train = []
            loss_D_A_train = []
            loss_D_B_train = []
            loss_cycle_A_train = []
            loss_cycle_B_train = []
            loss_identity_A_train = []
            loss_identity_B_train = []
            
            for _iter, data in enumerate(self.dataloader):
                real_A = data['img_A'].to(device=self.args.device)
                real_B = data['img_B'].to(device=self.args.device)

                fake_B = self.netG_A2B(real_A)
                fake_A = self.netG_B2A(real_B)

                recon_A = self.netG_B2A(fake_B)
                recon_B = self.netG_A2B(fake_A)

                identity_A = self.netG_B2A(real_A)
                identity_B = self.netG_A2B(real_B)

                #Discriminator Loss
                set_requires_grad([self.netD_A, self.netD_B], True)
                self.optimizerD.zero_grad()
                
                real_A_dis = self.netD_A(real_A)
                fake_A_dis = self.netD_A(fake_A.detach())

                loss_D_A_real = self.criterion_GAN(real_A_dis, torch.ones_like(real_A_dis))
                loss_D_A_fake = self.criterion_GAN(fake_A_dis, torch.zeros_like(fake_A_dis))
                loss_D_A = 0.5 * (loss_D_A_real + loss_D_A_fake)

                real_B_dis = self.netD_B(real_B)
                fake_B_dis = self.netD_B(fake_B.detach())

                loss_D_B_real = self.criterion_GAN(real_B_dis, torch.ones_like(real_B_dis))
                loss_D_B_fake = self.criterion_GAN(fake_B_dis, torch.zeros_like(fake_B_dis))
                loss_D_B = 0.5 * (loss_D_B_real + loss_D_B_fake)

                loss_D = loss_D_A + loss_D_A
                loss_D.backward()

                set_requires_grad([self.netD_A, self.netD_B], False)
                self.optimizerG.zero_grad()
                
                #Generator loss
                fake_A_dis = self.netD_A(fake_A)
                fake_B_dis = self.netD_B(fake_B)
                # Adversarial Loss
                loss_G_A2B = self.criterion_GAN(fake_A_dis, torch.ones_like(fake_A_dis))
                loss_G_B2A = self.criterion_GAN(fake_B_dis, torch.ones_like(fake_B_dis))
                # cycle consistancy loss
                loss_cycle_A = self.criterion_cycle(recon_A, real_A) * 10
                loss_cycle_B = self.criterion_cycle(recon_B, real_B) * 10
                # itentity loss
                loss_identity_A = self.criterion_identity(real_A, identity_A) * 5
                loss_identity_B = self.criterion_identity(real_B, identity_B) * 5

                loss_G = loss_G_A2B + loss_G_B2A + loss_cycle_A + loss_cycle_B + loss_identity_A + loss_identity_B
                loss_G.backward()

                self.optimizerG.step()

                loss_G_A2B_train += [loss_G_A2B.item()]
                loss_G_B2A_train += [loss_G_B2A.item()]
                
                loss_D_A_train += [loss_D_A.item()]
                loss_D_B_train += [loss_D_B.item()]
                
                loss_cycle_A_train += [loss_cycle_A.item()]
                loss_cycle_B_train += [loss_cycle_B.item()]
                
                loss_identity_A_train += [loss_identity_A.item()]
                loss_identity_B_train += [loss_identity_B.item()]

               
                print("Train : Epoch %04d/ %04d | Batch %04d / %04d | "
                       "Generator A2B %.4f B2A %.4f | "
                       "Discriminator A %.4f B %.4f | "
                       "Cycle A %.4f B %.4f | "
                       "Identity A %.4f B %.4f | " % 
                       (epoch, self.args.num_epochs, _iter, len(self.dataloader),
                       np.mean(loss_G_A2B_train), np.mean(loss_G_B2A_train),
                       np.mean(loss_D_A_train), np.mean(loss_D_B_train),
                       np.mean(loss_cycle_A_train), np.mean(loss_cycle_B_train),
                       np.mean(loss_identity_A_train), np.mean(loss_identity_B_train)))
            
            save(ckpt_path, self.netG_A2B, self.netG_B2A, self.netD_A, self.netD_B, self.optimizerG, self.optimizerD, epoch)
        
            self.lr_scheduler_G.step()
            self.lr_scheduler_D.step()
        
        