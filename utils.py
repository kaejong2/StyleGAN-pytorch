import random
import time
import datetime
import sys
import os
from torch.nn import init
from torch.autograd import Variable
import torch
import numpy as np
from torch.optim import lr_scheduler

def init_weight(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def save(ckpt_dir, netG_A2B, netG_B2A, netD_A, netD_B, optimG, optimD, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    torch.save({'netG_A2B' : netG_A2B.state_dict(), 'netG_B2A' : netG_B2A.state_dict(), 'netD_A':netD_A.state_dict(), 'netD_B': netD_B.state_dict(), 'optimG':optimG.state_dict(), 'optimD': optimD.state_dict()},
    "%s/model_epoch%d.pth" % (ckpt_dir,epoch))

def load(ckpt_dir, netG_A2B, netG_B2A, netD_A, netD_B, optimG, optimD):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return netG_A2B, netG_B2A, netD_A, netD_B, optimG, optimD, epoch

    device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst = [f for f in ckpt_lst if f.endswich('pth')]
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.indigit, f))))

    dict_model = torch.load('%s/%s'% (ckpt_dir, ckpt_lst[-1]), map_location=device)

    netG_A2B.load_state_dict(dict_model['netG_A2B'])
    netG_B2A.load_state_dict(dict_model['netG_B2A'])
    netD_A.load_state_dict(dict_model['netD_A'])
    netD_B.load_state_dict(dict_model['netD_B'])
    optimG.load_state_dict(dict_model['optimG'])
    optimD.load_state_dict(dict_model['optimD'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return netG_A2B, netG_B2A, netD_A, netD_B, optimG, optimD, epoch

def load_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path)
    print('Loading checkpoint from %s succeed' % ckpt_path)
    return ckpt


def set_requires_grad(nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

class LambdaLR():
    def __init__(self, epochs, offset, decay_epoch):
        self.epochs = epochs
        self.offset = offset
        self.decay_epoch = decay_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_epoch)/(self.epochs - self.decay_epoch)


