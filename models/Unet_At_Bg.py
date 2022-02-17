import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import AttentionModule

class UnetAtGenerator(nn.Module):

    # initializers
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()
        d = ngf
        use_bias = False

        # Unet encoder
        self.conv1 = nn.Conv2d(input_nc, d, 4, 2, 1,bias=use_bias)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1,bias=use_bias)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1,bias=use_bias)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1,bias=use_bias)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, d * 8, 4, 2, 1,bias=use_bias)
        self.conv5_bn = nn.BatchNorm2d(d * 8)
        self.conv6 = nn.Conv2d(d * 8, d * 8, 4, 2, 1,bias=use_bias)
        self.conv6_bn = nn.BatchNorm2d(d * 8)
        self.conv7 = nn.Conv2d(d * 8, d * 8, 4, 2, 1,bias=use_bias)
        self.conv7_bn = nn.BatchNorm2d(d * 8)
        self.conv8 = nn.Conv2d(d * 8, d * 8, 4, 2, 1,bias=use_bias)

        # Unet decoder
        self.deconv1 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1,bias=use_bias)
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1,bias=use_bias)
        self.deconv2_bn = nn.BatchNorm2d(d * 8)
        self.deconv3 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1,bias=use_bias)
        self.deconv3_bn = nn.BatchNorm2d(d * 8)
        self.deconv4 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1,bias=use_bias)
        self.deconv4_bn = nn.BatchNorm2d(d * 8)
        self.deconv5 = nn.ConvTranspose2d(d * 8 * 2, d * 4, 4, 2, 1,bias=use_bias)
        self.deconv5_bn = nn.BatchNorm2d(d * 4)
        self.deconv6 = nn.ConvTranspose2d(d * 4 * 2, d * 2, 4, 2, 1,bias=use_bias)
        self.deconv6_bn = nn.BatchNorm2d(d * 2)
        self.deconv7 = nn.ConvTranspose2d(d * 2 * 2, d, 4, 2, 1,bias=use_bias)
        self.deconv7_bn = nn.BatchNorm2d(d)
        self.deconv8 = nn.ConvTranspose2d(d * 2, output_nc, 4, 2, 1)

        self.at1 = AttentionModule(1024)
        self.at2 = AttentionModule(1024)
        self.at3 = AttentionModule(1024)

    # forward method
    def forward(self, input):

        e1 = self.conv1(input)
        e2 = self.conv2_bn(self.conv2(F.leaky_relu(e1, 0.2)))
        e3 = self.conv3_bn(self.conv3(F.leaky_relu(e2, 0.2)))
        e4 = self.conv4_bn(self.conv4(F.leaky_relu(e3, 0.2)))
        e5 = self.conv5_bn(self.conv5(F.leaky_relu(e4, 0.2)))
        e6 = self.conv6_bn(self.conv6(F.leaky_relu(e5, 0.2)))
        e7 = self.conv7_bn(self.conv7(F.leaky_relu(e6, 0.2)))

        e8 = self.conv8(F.leaky_relu(e7, 0.2))

        d1 = F.dropout(self.deconv1_bn(self.deconv1(F.relu(e8))), 0.5, training=True)
        d1 = torch.cat([d1, e7], 1)

        ## add attention module on d1, d2, d3
        at_d1 = self.at1(d1)
        
        d2 = F.dropout(self.deconv2_bn(self.deconv2(F.relu(at_d1))), 0.5, training=True)
        d2 = torch.cat([d2, e6], 1)
        at_d2 = self.at2(d2)

        d3 = F.dropout(self.deconv3_bn(self.deconv3(F.relu(at_d2))), 0.5, training=True)
        d3 = torch.cat([d3, e5], 1)
        at_d3 = self.at3(d3)

        d4 = self.deconv4_bn(self.deconv4(F.relu(at_d3)))
        d4 = torch.cat([d4, e4], 1)

        d5 = self.deconv5_bn(self.deconv5(F.relu(d4)))
        d5 = torch.cat([d5, e3], 1)

        d6 = self.deconv6_bn(self.deconv6(F.relu(d5)))
        d6 = torch.cat([d6, e2], 1)
        
        d7 = self.deconv7_bn(self.deconv7(F.relu(d6)))
        d7 = torch.cat([d7, e1], 1)
        
        d8 = self.deconv8(F.relu(d7))
        o = F.tanh(d8)
        
        return o

class UnetAtBgGenerator(nn.Module):

    # initializers
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()
        d = ngf
        use_bias = False

        # Unet encoder
        self.conv1 = nn.Conv2d(input_nc, d, 4, 2, 1,bias=use_bias)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1,bias=use_bias)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1,bias=use_bias)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1,bias=use_bias)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, d * 8, 4, 2, 1,bias=use_bias)
        self.conv5_bn = nn.BatchNorm2d(d * 8)
        self.conv6 = nn.Conv2d(d * 8, d * 8, 4, 2, 1,bias=use_bias)
        self.conv6_bn = nn.BatchNorm2d(d * 8)
        self.conv7 = nn.Conv2d(d * 8, d * 8, 4, 2, 1,bias=use_bias)
        self.conv7_bn = nn.BatchNorm2d(d * 8)
        self.conv8 = nn.Conv2d(d * 8, d * 8, 4, 2, 1,bias=use_bias)

        # Unet decoder
        self.deconv1 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1,bias=use_bias)
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1,bias=use_bias)
        self.deconv2_bn = nn.BatchNorm2d(d * 8)
        self.deconv3 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1,bias=use_bias)
        self.deconv3_bn = nn.BatchNorm2d(d * 8)
        self.deconv4 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1,bias=use_bias)
        self.deconv4_bn = nn.BatchNorm2d(d * 8)
        self.deconv5 = nn.ConvTranspose2d(d * 8 * 2, d * 4, 4, 2, 1,bias=use_bias)
        self.deconv5_bn = nn.BatchNorm2d(d * 4) # 256
        self.deconv6 = nn.ConvTranspose2d(d * 4 * 2, d * 2, 4, 2, 1,bias=use_bias)
        self.deconv6_bn = nn.BatchNorm2d(d * 2) # 128
        self.deconv7 = nn.ConvTranspose2d(d * 2 * 2, d, 4, 2, 1,bias=use_bias)
        self.deconv7_bn = nn.BatchNorm2d(d) # 64

        self.deconv_8 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv8_bn = nn.BatchNorm2d(d)

        self.conv_final = nn.Conv2d(d, output_nc, 3, padding=1,bias=use_bias)

        self.at1 = AttentionModule(1024)
        self.at2 = AttentionModule(1024)
        self.at3 = AttentionModule(1024)

        self.bg_encoder = BgEncoder(ngf=ngf,use_bias=use_bias)

    # forward method
    def forward(self, input, img=None, matte=None, noise=None):
        
        e1 = self.conv1(input)
        e2 = self.conv2_bn(self.conv2(F.leaky_relu(e1, 0.2)))
        e3 = self.conv3_bn(self.conv3(F.leaky_relu(e2, 0.2)))
        e4 = self.conv4_bn(self.conv4(F.leaky_relu(e3, 0.2)))
        e5 = self.conv5_bn(self.conv5(F.leaky_relu(e4, 0.2)))
        e6 = self.conv6_bn(self.conv6(F.leaky_relu(e5, 0.2)))
        e7 = self.conv7_bn(self.conv7(F.leaky_relu(e6, 0.2)))
        e8 = self.conv8(F.leaky_relu(e7, 0.2))

        d1 = F.dropout(self.deconv1_bn(self.deconv1(F.relu(e8))), 0.5, training=True)
        d1 = torch.cat([d1, e7], 1)

        ## add attention module on d1, d2, d3
        at_d1 = self.at1(d1)
        
        d2 = F.dropout(self.deconv2_bn(self.deconv2(F.relu(at_d1))), 0.5, training=True)
        d2 = torch.cat([d2, e6], 1)
        at_d2 = self.at2(d2)

        d3 = F.dropout(self.deconv3_bn(self.deconv3(F.relu(at_d2))), 0.5, training=True)
        d3 = torch.cat([d3, e5], 1)
        at_d3 = self.at3(d3)

        bg_feats, hair_mattes = self.bg_encoder(img, matte, noise)

        d4 = self.deconv4_bn(self.deconv4(F.relu(at_d3)))
        d4 = torch.cat([d4, e4], 1)
        
        d5 = self.deconv5_bn(self.deconv5(F.relu(d4)))
        d5 = torch.cat([d5, e3], 1)
        d5 = d5 * hair_mattes[0] + bg_feats[0] * (1 - hair_mattes[0])

        d6 = self.deconv6_bn(self.deconv6(F.relu(d5)))
        d6 = torch.cat([d6, e2], 1)
        d6 = d6 * hair_mattes[1] + bg_feats[1] * (1 - hair_mattes[1])
        
        d7 = self.deconv7_bn(self.deconv7(F.relu(d6)))
        d7 = torch.cat([d7, e1], 1)
        d7 = d7 * hair_mattes[2] + bg_feats[2] * (1 - hair_mattes[2])

        d_8 = self.deconv8_bn(self.deconv_8(F.relu(d7)))
        d_8 = d_8 * hair_mattes[3] + bg_feats[3] * (1 - hair_mattes[3])

        final = self.conv_final(F.relu(d_8))
        o_f = torch.tanh(final)
        
        return o_f

class BgEncoder(nn.Module):
    def __init__(self, ngf,use_bias=False):
        super().__init__()
        self.ngf = ngf
        self.conv1 = ConvBlock(3, self.ngf, 7, 1, 3, norm='none', activation='relu', pad_type='reflect')
        self.layer1 = ConvBlock(self.ngf, 2 * self.ngf, 4, 2, 1, norm='bn', activation='relu', pad_type='reflect', use_bias=use_bias)
        self.layer2 = ConvBlock(2 * self.ngf, 4 * self.ngf, 4, 2, 1, norm='bn', activation='relu', pad_type='reflect', use_bias=use_bias)
        self.layer3 = ConvBlock(4 * self.ngf, 8 * self.ngf, 4, 2, 1, norm='bn', activation='relu', pad_type='reflect', use_bias=use_bias)

    def forward(self, image, mask, noise):
        hair_matte = torch.unsqueeze(mask[:, 0, :, :], 1)

        input = image * (1 - hair_matte) + noise * hair_matte

        x0 = self.conv1(input) # 64
        x1 = self.layer1(x0) # 1/2 64*2
        x2 = self.layer2(x1) # 1/4 64*4
        x3 = self.layer3(x2) # 1/8 64*8

        _,_,sh,sw = hair_matte.size()
        hair_matte1 = F.interpolate(hair_matte, size=(int(sh/2), int(sw/2)), mode='nearest')
        hair_matte2 = F.interpolate(hair_matte, size=(int(sh / 4), int(sw / 4)), mode='nearest')
        hair_matte3 = F.interpolate(hair_matte, size=(int(sh / 8), int(sw / 8)), mode='nearest')

        return [x3, x2, x1, x0], [hair_matte3, hair_matte2, hair_matte1, hair_matte]

class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero',use_bias = True):
        super(ConvBlock, self).__init__()
        self.use_bias = use_bias
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x






