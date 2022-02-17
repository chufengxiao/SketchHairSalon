import torch, torchvision
from torch import nn
from torch.nn import Softmax
from .base_model import BaseModel
from . import networks
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

class Pix2PixHairModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch', netG='unet_at', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument("--is_braid",action="store_true")
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_LS', type=float, default=100.0, help='weight for LS loss')
            parser.add_argument('--lambda_blur', type=float, default=100.0, help='weight for Smooth loss')
            
        parser.add_argument("--visual_only_fake",action="store_true")
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1','G_style', 'D_real', 'D_fake']

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        if opt.visual_only_fake:
            self.visual_names = ['fake_B']
        else:
            self.visual_names = ['real_A', 'fake_B', 'real_B']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # self.criterionStyle = VGGLoss(opt.gpu_ids) 
            self.criterionStyle = networks.VGGPerceptualLoss(opt.gpu_ids)

            if opt.is_braid:
                self.smoothing = networks.GaussianSmoothing(opt.output_nc, 10, 10).to(self.device)
                self.loss_names.append('G_B')

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            
            

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device) if input['B'] is not None else None

        self.noise = input['N'].to(self.device)
        self.matte = input['M'].to(self.device)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        

    def forward(self):
        self.fake_B = self.netG(self.real_A,self.real_B,self.matte,self.noise)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):

        fake_AB = torch.cat((self.real_A, self.fake_B), 1)

        # GAN loss
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # L1 Loss
        G_L1= self.criterionL1(self.fake_B, self.real_B)
        self.loss_G_L1 = G_L1 * self.opt.lambda_L1

        # Perceptual loss
        G_style = self.criterionStyle(self.fake_B, self.real_B)
        self.loss_G_style = G_style * self.opt.lambda_LS

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_style

        # Shape loss for braided hairstyle
        if self.opt.is_braid:
            s_matte = F.interpolate(self.matte, size=(503,503), mode='nearest')
            s_fake = s_matte*self.smoothing((self.fake_B+1)/2)
            s_real = s_matte*self.smoothing((self.real_B+1)/2)
            self.loss_G_B = self.opt.lambda_blur * self.criterionL1(s_fake, s_real)
            self.loss_G += self.loss_G_B

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

