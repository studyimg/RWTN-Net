import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_blocks import CBAM

def _upsample_like(src,tar):
    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src

class GlobalExtraction(nn.Module):
    def __init__(self,dim = None):
        super().__init__()
        self.avgpool = self.globalavgchannelpool
        self.maxpool = self.globalmaxchannelpool
        self.proj = nn.Sequential(
            nn.Conv2d(2, 1, 1,1),
            nn.BatchNorm2d(1)
        )
    def globalavgchannelpool(self, x):
        x = x.mean(1, keepdim = True)
        return x

    def globalmaxchannelpool(self, x):
        x = x.max(dim = 1, keepdim=True)[0]
        return x

    def forward(self, x):
        x_ = x.clone()
        x = self.avgpool(x)
        x2 = self.maxpool(x_)

        cat = torch.cat((x,x2), dim = 1)

        proj = self.proj(cat)
        return proj

class ContextExtraction(nn.Module):
    def __init__(self, dim, reduction = None):
        super().__init__()
        self.reduction = 1 if reduction == None else 2

        self.dconv = self.DepthWiseConv2dx2(dim)
        self.proj = self.Proj(dim)

    def DepthWiseConv2dx2(self, dim):
        dconv = nn.Sequential(
            nn.Conv2d(in_channels = dim,
                      out_channels = dim,
                      kernel_size = 3,
                      padding = 1,
                      groups = dim),
            nn.BatchNorm2d(num_features = dim),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = dim,
                      out_channels = dim,
                      kernel_size = 3,
                      padding = 2,
                      dilation = 2),
            nn.BatchNorm2d(num_features = dim),
            nn.ReLU(inplace = True)
        )
        return dconv

    def Proj(self, dim):
        proj = nn.Sequential(
            nn.Conv2d(in_channels = dim,
                      out_channels = dim //self.reduction,
                      kernel_size = 1
                      ),
            nn.BatchNorm2d(num_features = dim//self.reduction)
        )
        return proj
    def forward(self,x):
        x = self.dconv(x)
        x = self.proj(x)
        return x

class MultiscaleFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.local= ContextExtraction(dim)
        self.global_ = GlobalExtraction()
        self.bn = nn.BatchNorm2d(num_features=dim)

    def forward(self, x, g,):
        x = self.local(x)
        g = self.global_(g)

        fuse = self.bn(x + g)
        return fuse

class MultiScaleGatedAttn(nn.Module):
    # Version 1
    def __init__(self, dim):
        super().__init__()
        self.multi = MultiscaleFusion(dim)
        self.selection = nn.Conv2d(dim, 2,1)
        self.proj = nn.Conv2d(dim, dim,1)
        self.bn = nn.BatchNorm2d(dim)
        self.bn_2 = nn.BatchNorm2d(dim)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim,
                      kernel_size=1, stride=1))

    def forward(self,x,g):
        x_ = x.clone()
        g_ = g.clone()



        multi = self.multi(x, g) # B, C, H, W

        ### Option 2 ###
        multi = self.selection(multi) # B, num_path, H, W

        attention_weights = F.softmax(multi, dim=1)  # Shape: [B, 2, H, W]

        A, B = attention_weights.split(1, dim=1)  # Each will have shape [B, 1, H, W]

        x_att = A.expand_as(x_) * x_  # Using expand_as to match the channel dimensions
        g_att = B.expand_as(g_) * g_

        x_att = x_att + x_
        g_att = g_att + g_

        x_sig = torch.sigmoid(x_att)
        g_att_2 = x_sig * g_att

        g_sig = torch.sigmoid(g_att)
        x_att_2 = g_sig * x_att

        interaction = x_att_2 * g_att_2

        projected = torch.sigmoid(self.bn(self.proj(interaction)))

        weighted = projected * x_

        y = self.conv_block(weighted)

        y = self.bn_2(y)
        return y


class PConv2d_(nn.Module):
    def __init__(self,
                 in_channels,
                 kernel_size=3,
                 dilation=1,
                 n_div: int = 2):
        super(PConv2d_, self).__init__()
        self.dim_conv = in_channels // n_div
        self.dim_untouched = in_channels - self.dim_conv

        self.conv = nn.Conv2d(in_channels=self.dim_conv,
                              out_channels=self.dim_conv,
                              kernel_size=kernel_size,
                              stride=1,
                              dilation=dilation,
                              padding=dilation,
                              bias=False)
        self.CBAM = CBAM(in_channels)


    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.conv(x1)
        x = torch.cat((x1, x2), dim=1)
        x = self.CBAM(x)

        return x

#逐点卷积加BN和激活函数
class PWConv(nn.Module):
    def __init__(self, in_planes, out_planes, act=None,kernel_size=1, stride=1, padding=0, dilation=1, groups=1):
        super(PWConv, self).__init__()
        self.act_ = act
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
        if act == None:
            self.bn  = nn.Identity()
            self.act = nn.Identity()  #代表该层不做操作
        else:
            self.bn = nn.BatchNorm2d(out_planes)
            if act == 'gelu':
                self.act = nn.GELU()
            elif act == 'relu':
                self.act = nn.ReLU()


    def forward(self, x):
        x = self.conv(x)
        if self.act_ != None:
            x = self.bn(x)
            x = self.act(x)
        return x


# 输出和输入通道数一致 进行pconv和两次逐点卷积，第一次逐点有激活函数
class PAConv_dil(nn.Module):
    def __init__(self,
                 in_channels: int,
                 dilation : int,
                 act: str = 'gelu',
                 mid = 2
                 ):
        super(PAConv_dil, self).__init__()

        self.conv1 = PConv2d_(in_channels,dilation=dilation)#不影响尺寸
        self.conv2 = PWConv(in_channels,in_channels*mid,act=act)
        self.conv3 = PWConv(in_channels*mid,in_channels)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)

        return x + y




class AUB4(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12):
        super(AUB4,self).__init__()

        self.rebnconvin = PWConv(in_ch,mid_ch,act='gelu',kernel_size=3,padding=1)
        self.rebnconv1 = PAConv_dil(mid_ch,dilation=1)
        self.rebnconv2 = PAConv_dil(mid_ch,dilation=2)
        self.rebnconv3 = PAConv_dil(mid_ch,dilation=4)
        self.rebnconv4 = PAConv_dil(mid_ch,dilation=8)
        self.rebnconv3d = PWConv(mid_ch*2,mid_ch,act='gelu',kernel_size=3,dilation=4,padding=4)
        self.rebnconv2d = PWConv(mid_ch*2,mid_ch,act='gelu',kernel_size=3,dilation=2,padding=2)
        self.rebnconv1d = PWConv(mid_ch*2,mid_ch,act='gelu',kernel_size=3,dilation=1,padding=1)

    def forward(self,x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx2d = self.rebnconv2d(torch.cat((hx3d,hx2),1))
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))

        return hx1d + hxin






class AUB6(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(AUB6,self).__init__()

        self.rebnconvin = PWConv(in_ch,mid_ch,act='gelu',kernel_size=3,padding=1)
        self.rebnconv1 = PAConv_dil(mid_ch,dilation=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv2 = PAConv_dil(mid_ch,dilation=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv3 = PAConv_dil(mid_ch,dilation=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv4 = PAConv_dil(mid_ch,dilation=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv5 = PAConv_dil(mid_ch,dilation=1)
        self.rebnconv6 = PAConv_dil(mid_ch,dilation=2)#输出和输入通道数一致，尺寸也不变
        self.rebnconv5d = PWConv(mid_ch,mid_ch,act='gelu',kernel_size=3,dilation=1,padding=1)
        self.rebnconv4d = PWConv(mid_ch,mid_ch,act='gelu',kernel_size=3,dilation=1,padding=1)
        self.rebnconv3d = PWConv(mid_ch,mid_ch,act='gelu',kernel_size=3,dilation=1,padding=1)
        self.rebnconv2d = PWConv(mid_ch,mid_ch,act='gelu',kernel_size=3,dilation=1,padding=1)
        self.rebnconv1d = PWConv(mid_ch,mid_ch,act='gelu',kernel_size=3,dilation=1,padding=1)

        self.MultiScaleGatedAttn1=MultiScaleGatedAttn(mid_ch)
        self.MultiScaleGatedAttn2=MultiScaleGatedAttn(mid_ch)
        self.MultiScaleGatedAttn3=MultiScaleGatedAttn(mid_ch)
        self.MultiScaleGatedAttn4=MultiScaleGatedAttn(mid_ch)
        self.MultiScaleGatedAttn5=MultiScaleGatedAttn(mid_ch)



    def forward(self,x):

        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx6 = self.rebnconv6(hx5)
        hx5d =  self.rebnconv5d(self.MultiScaleGatedAttn5(hx6,hx5))
        hx5dup = _upsample_like(hx5d,hx4)
        hx4d = self.rebnconv4d(self.MultiScaleGatedAttn4(hx5dup,hx4))
        hx4dup = _upsample_like(hx4d,hx3)
        hx3d = self.rebnconv3d(self.MultiScaleGatedAttn3(hx4dup,hx3))
        hx3dup = _upsample_like(hx3d,hx2)
        hx2d = self.rebnconv2d(self.MultiScaleGatedAttn2(hx3dup,hx2))
        hx2dup = _upsample_like(hx2d,hx1)
        hx1d = self.rebnconv1d(self.MultiScaleGatedAttn1(hx2dup,hx1))
        return hx1d + hxin