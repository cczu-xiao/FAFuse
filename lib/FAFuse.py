import torch
import torch.nn as nn
from torchvision.models import resnet34 as resnet
from .DeiT import deit_small_patch16_224 as deit
from .utils import *
import torch.nn.functional as F
import math


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, bias=False,padding=1)

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=True)

class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention_dynamic, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Priority on encoding

        ## Initial values

        self.f_qr = nn.Parameter(torch.tensor(0.1),  requires_grad=False)
        self.f_kr = nn.Parameter(torch.tensor(0.1),  requires_grad=False)
        self.f_sve = nn.Parameter(torch.tensor(0.1),  requires_grad=False)
        self.f_sv = nn.Parameter(torch.tensor(1.0),  requires_grad=False)


        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()
        # self.print_para()

    def forward(self, x):
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H), [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2, self.kernel_size, self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=0)
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)


        # multiply by factors
        qr = torch.mul(qr, self.f_qr)
        kr = torch.mul(kr, self.f_kr)

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)

        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)

        # multiply by factors
        sv = torch.mul(sv, self.f_sv)
        sve = torch.mul(sve, self.f_sve)

        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output
    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))


class FAFusion_block(nn.Module):
    def __init__(self, ch_1, ch_2,  ch_int, ch_out, drop_rate=0., scale_size=1):
        super(FAFusion_block, self).__init__()
        self.scale_size = scale_size
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        # AG anxial attention for F_l
        self.conv_down = nn.Conv2d(ch_int, ch_int, kernel_size=3, stride=1, bias=True, padding=1,groups=1 )
        self.conv_down1 = conv1x1(ch_1 + ch_2, ch_int)
        self.conv_up1 = nn.Conv2d(ch_int * 2, ch_int,kernel_size=1,stride=1,padding=0,bias=True)
        self.bn1 = nn.BatchNorm2d(ch_int)
        self.hight_block = AxialAttention(ch_int, ch_int, groups=4, kernel_size=22 * self.scale_size)
        self.width_block = AxialAttention(ch_int, ch_int, groups=4, kernel_size=22 * self.scale_size,stride=1, width=True)
        self.diagonal_block1 = AxialAttention(ch_int, ch_int, groups=4,kernel_size=22 * self.scale_size,stride=1)
        self.diagonal_block2 = AxialAttention(ch_int, ch_int, groups=4, kernel_size=22 * self.scale_size+1,stride=1)
        self.conv_up = conv1x1(ch_int, ch_int)
        self.bn2 = nn.BatchNorm2d(ch_int)

        # bi-linear modelling for both
        self.W_g = Conv(ch_1, ch_int, 1, bn=True, relu=False)  
        self.W_x = Conv(ch_2, ch_int, 1, bn=True, relu=False)  
        self.W = Conv(ch_int, ch_int, 3, bn=True, relu=True) 
        self.residual = Residual(ch_int*2, ch_out)
        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate
        self.convpro = ConvPro(ch_1 + ch_2,ch_1 + ch_2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.LReLU = nn.LeakyReLU(0.05, True)

    def forward(self, g, x): 

        W_g = self.W_g(g)
        W_x = self.W_x(x)  
        bp_m = self.W(W_g * W_x)  

        bp = torch.cat([g,x],dim=1)
        # MSC
        bp = self.convpro(bp)

        bp = self.conv_down1(bp)    
        bp = self.bn1(bp)
        bp = self.relu(bp)

        bp_in = bp

        # H&W
        bp_1 = bp
        bp = self.hight_block(bp)
        bp = self.width_block(bp)
        bp = bp + bp_1
        bp = self.conv_down(bp)

        bp_in2 = bp
        
        
        # M&C
        bp_1 = bp
        bp = F.pad(bp, (0, 0, 0, 1), value=0)
        bp = bp.reshape(*bp.size()[:-2], bp.size(-1), bp.size(-2))
        bp = self.diagonal_block1(bp)
        bp = bp.reshape(*bp.size()[:-2], bp.size(-1), bp.size(-2))
        bp = F.pad(bp, (0, 0, 0, -1), value=0)

        
        bp = F.pad(bp, (0, 1, 0, 0), value=0)
        bp = bp.reshape(*bp.size()[:-2], bp.size(-1), bp.size(-2))
        bp = self.diagonal_block2(bp)
        bp = bp.reshape(*bp.size()[:-2], bp.size(-1), bp.size(-2))
        bp = F.pad(bp, (0, -1, 0, 0), value=0)

        bp = bp + bp_1

        bp = self.conv_down(bp)

        bp = torch.cat([bp_in2,bp],dim=1)
        bp = self.conv_up1(bp)
        bp = self.LReLU(bp)

        bp = bp + bp_in

        fuse = self.residual(torch.cat([bp_m,bp], 1))

        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse

class FAFuse_B(nn.Module):
    def __init__(self, num_classes=1, drop_rate=0.2, normal_init=True, pretrained=False):
        super(FAFuse_B, self).__init__()

        self.resnet = resnet()
        if pretrained:
            self.resnet.load_state_dict(torch.load('pretrained/resnet34-43635321.pth'))
        self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()
        self.transformer = deit(pretrained=pretrained)
        self.up1 = Up(in_ch1=768, out_ch=128)
        self.up2 = Up(128, 64)

        self.final_x = nn.Sequential(
            Conv(256, 64, 1, bn=True, relu=True),
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )

        self.final_1 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )

        self.final_2 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )
        self.up_c = FAFusion_block(ch_1=256, ch_2=768, ch_int=256, ch_out=256, drop_rate=drop_rate / 2, scale_size=1)
        self.up_c_1_1 = FAFusion_block(ch_1=128, ch_2=128, ch_int=128, ch_out=128, drop_rate=drop_rate / 2, scale_size=2)
        self.up_c_1_2 = Up(in_ch1=256, out_ch=128, in_ch2=128, attn=True)
        self.up_c_2_1 = FAFusion_block(ch_1=64, ch_2=64, ch_int=64, ch_out=64, drop_rate=drop_rate / 2, scale_size=4)
        self.up_c_2_2 = Up(128, 64, 64, attn=True)

        self.drop = nn.Dropout2d(drop_rate)

        if normal_init:
            self.init_weights()


    def forward(self, imgs, labels=None):
        # Transformer Branch
        x_b = self.transformer(imgs)
        x_b = torch.transpose(x_b, 1, 2)
        x_b = x_b.view(x_b.shape[0], -1, 22, 22)
        x_b = self.drop(x_b)

        x_b_1 = self.up1(x_b)
        x_b_1 = self.drop(x_b_1)
        x_b_2 = self.up2(x_b_1)
        x_b_2 = self.drop(x_b_2)

        # CNN Branch
        x_u = self.resnet.conv1(imgs)
        x_u = self.resnet.bn1(x_u)
        x_u = self.resnet.relu(x_u)

        x_u = self.resnet.maxpool(x_u)

        x_u_2 = self.resnet.layer1(x_u)
        x_u_2 = self.drop(x_u_2)
        x_u_1 = self.resnet.layer2(x_u_2)
        x_u_1 = self.drop(x_u_1)
        x_u = self.resnet.layer3(x_u_1)
        x_u = self.drop(x_u)

        # joint Branch
        x_c = self.up_c(x_u, x_b)

        x_c_1_1 = self.up_c_1_1(x_u_1, x_b_1)
        x_c_1 = self.up_c_1_2(x_c, x_c_1_1)

        x_c_2_1 = self.up_c_2_1(x_u_2, x_b_2)
        x_c_2 = self.up_c_2_2(x_c_1, x_c_2_1)

        # decoder part
        map_x = F.interpolate(self.final_x(x_c), scale_factor=16, mode='bilinear', align_corners=True)
        map_1 = F.interpolate(self.final_1(x_b_2), scale_factor=4, mode='bilinear', align_corners=True)
        map_2 = F.interpolate(self.final_2(torch.add(x_c_2,x_u_2)), scale_factor=4, mode='bilinear', align_corners=True)
        return map_x, map_1, map_2

    def init_weights(self):
        self.up1.apply(init_weights)
        self.up2.apply(init_weights)
        self.final_x.apply(init_weights)
        self.final_1.apply(init_weights)
        self.final_2.apply(init_weights)
        self.up_c.apply(init_weights)
        self.up_c_1_1.apply(init_weights)
        self.up_c_1_2.apply(init_weights)
        self.up_c_2_1.apply(init_weights)
        self.up_c_2_2.apply(init_weights)

def init_weights(m):
    """
    Initialize weights of layers using Kaiming Normal (He et al.) as argument of "Apply" function of
    "nn.Module"
    :param m: Layer to initialize
    :return: None
    """
    if isinstance(m, nn.Conv2d):
        '''
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        trunc_normal_(m.weight, std=math.sqrt(1.0/fan_in)/.87962566103423978)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        '''
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch1 + in_ch2, out_ch)

        if attn:
            self.attn_block = Attention_block(in_ch1, in_ch2, out_ch)
        else:
            self.attn_block = None

    def forward(self, x1, x2=None):

        x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

            if self.attn_block is not None:
                x2 = self.attn_block(x1, x2)
            x1 = torch.cat([x2, x1], dim=1)
        x = x1
        return self.conv(x)

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,groups=out_channels),
            nn.BatchNorm2d(out_channels)
        )
        self.identity = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.double_conv(x) + self.identity(x))

class ConvPro(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_1 = conv(in_channels, out_channels // 4, kernel_size=3, padding=1,
                           stride=1, groups=1)
        self.conv_2 = conv(in_channels, out_channels // 4, kernel_size=5, padding=2,
                           stride=1, groups=4)
        self.conv_3 = conv(in_channels, out_channels // 4, kernel_size=7, padding=3,
                           stride=1, groups=8)
        self.conv_4 = conv(in_channels, out_channels // 4, kernel_size=9, padding=4,
                           stride=1, groups=16)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)
        x = torch.cat((x1,x2,x3,x4),dim=1)
        return x

class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)  #

        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out) + out
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
