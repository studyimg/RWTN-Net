import torch
import torch.nn as nn
import torch.nn.functional as F
from AUB import AUB4,AUB6
from FRFE import FRFE
from SConv import SConv_2d


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class ASFF2(nn.Module):
    def __init__(self, level, multiplier=1, rfb=False, vis=False,dim=None):

        super(ASFF2, self).__init__()
        self.level = level
        self.dim = dim
        # print(self.dim)

        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = Conv(int(self.dim[1] * multiplier), self.inter_dim, 3, 2)

            self.expand = Conv(self.inter_dim, int(
                self.dim[0] * multiplier), 3, 1)
        elif level == 1:
            self.compress_level_0 = Conv(
                int(self.dim[0] * multiplier), self.inter_dim, 1, 1)

            self.expand = Conv(self.inter_dim, int(
                self.dim[1] * multiplier), 3, 1)



        compress_c = 8 if rfb else 16
        self.weight_level_0 = Conv(
            self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(
            self.inter_dim, compress_c, 1, 1)

        self.weight_levels = Conv(
            compress_c * 2, 2, 1, 1)
        self.vis = vis

    def forward(self, x):

        x_level_0 = x[1]  # 最大特征层
        x_level_1 = x[0]  # 最小特征层


        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)

        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=2, mode='bilinear')
            level_1_resized = x_level_1

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)


        levels_weight_v = torch.cat(
            (level_0_weight_v, level_1_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out

class ASFF3(nn.Module):
    def __init__(self, level, multiplier=1, rfb=False, vis=False, dim=None):

        super(ASFF3, self).__init__()
        self.level = level
        self.dim = dim


        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = Conv(int(self.dim[1] * multiplier), self.inter_dim, 3, 2)

            self.stride_level_2 = Conv(int(self.dim[2] * multiplier), self.inter_dim, 3, 2)

            self.expand = Conv(self.inter_dim, int(
                self.dim[0] * multiplier), 3, 1)
        elif level == 1:
            self.compress_level_0 = Conv(
                int(self.dim[0] * multiplier), self.inter_dim, 1, 1)
            self.stride_level_2 = Conv(
                int(self.dim[2] * multiplier), self.inter_dim, 3, 2)
            self.expand = Conv(self.inter_dim, int(self.dim[1] * multiplier), 3, 1)
        elif level == 2:
            self.compress_level_0 = Conv(
                int(self.dim[0] * multiplier), self.inter_dim, 1, 1)
            self.compress_level_1 = Conv(
                int(self.dim[1] * multiplier), self.inter_dim, 1, 1)
            self.expand = Conv(self.inter_dim, int(
                self.dim[2] * multiplier), 3, 1)


        compress_c = 8 if rfb else 16
        self.weight_level_0 = Conv(
            self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(
            self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(
            self.inter_dim, compress_c, 1, 1)

        self.weight_levels = Conv(
            compress_c * 3, 3, 1, 1)
        self.vis = vis

    def forward(self, x):

        x_level_0 = x[2]  # 最大特征层
        x_level_1 = x[1]  # 中间特征层
        x_level_2 = x[0]  # 最小特征层

        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(
                x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=2, mode='bilinear')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=4, mode='bilinear')
            x_level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(
                x_level_1_compressed, scale_factor=2, mode='bilinear')
            level_2_resized = x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)

        levels_weight_v = torch.cat(
            (level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out
class ChannelAtt(nn.Module):
    def __init__(self,c1=64,c2=128,c3=256,c4=512):
        super(ChannelAtt, self).__init__()

        self.linear_c4 = nn.Conv2d(c4, 32, 1)
        self.linear_c3 = nn.Conv2d(c3, 32, 1)
        self.linear_c2 = nn.Conv2d(c2, 32, 1)
        self.linear_c1 = nn.Conv2d(c1, 32, 1)

        #AUB4
        self.up_linear_fuse = AUB4(128, 64)

        self.gated1 = nn.Sequential(nn.Conv2d(128,64,kernel_size=3,stride=2,padding=1,bias=False),nn.BatchNorm2d(64),nn.GELU(),
                                    nn.Conv2d(64, 64, kernel_size=1, stride=1),
                                    nn.Conv2d(64, 64, kernel_size=3, stride=2,padding=1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
                                    nn.Conv2d(64, 128, kernel_size=1, stride=1),
                                    torch.nn.AdaptiveMaxPool2d((1,1), return_indices=False),
                                    torch.nn.Sigmoid()
                                    )
        self.gated2 = nn.Sequential(nn.Conv2d(128,64,kernel_size=3,stride=2,padding=1,bias=False),nn.BatchNorm2d(64),nn.GELU(),
                                    nn.Conv2d(64, 64, kernel_size=1, stride=1),
                                    nn.Conv2d(64, 64, kernel_size=3, stride=2,padding=1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
                                    nn.Conv2d(64, 128, kernel_size=1, stride=1),
                                    torch.nn.AdaptiveAvgPool2d((1,1)),
                                    torch.nn.Sigmoid()
                                    )


        self.outconv = nn.Conv2d(64, 128, 2, stride=2, padding=0)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear')

        self.down2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.down4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.down8 = nn.AvgPool2d(kernel_size=8, stride=8)


    def forward(self, c1, c2, c3, c4):


        _c4 = self.linear_c4(c4)
        up_c4 = self.up8(_c4)
        down_c4 = _c4


        _c3 = self.linear_c3(c3)
        up_c3 = self.up4(_c3)
        down_c3 = self.down2(_c3)


        _c2 = self.linear_c2(c2)
        up_c2 = self.up2(_c2)
        down_c2 = self.down4(_c2)

        _c1 = self.linear_c1(c1)
        up_c1 = _c1
        down_c1 = self.down8(_c1)

        x = torch.cat([up_c4, up_c3, up_c2, up_c1], dim=1)

        gate1 = self.gated1(torch.cat([down_c4, down_c3, down_c2, down_c1], dim=1))



        x = x + x * gate1

        x = self.outconv(self.up_linear_fuse(x))


        return x


def get_topk(x, k=10, dim=-3):
    val, _ = torch.topk(x, k=k, dim=dim)
    return val


class LASCC(nn.Module):
    def __init__(self, topk=3, patch_size=2):
        super().__init__()

        self.topk = topk
        self.patch_size = patch_size
        self.store = {}
        self.conv = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 2), stride=(1, 2))
        self.conv_transpose = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=1, stride=1)
        self.alpha = nn.Parameter(torch.tensor(5., dtype=torch.float32))


    def forward(self, x):  # x (b,c,h,w)
        patch_size = self.patch_size
        b, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        num_patch_h = int(h / patch_size)
        num_patch_w = int(w / patch_size)
        num_patches = num_patch_w * num_patch_h
        patch_area = patch_size * patch_size
        assert num_patches == (h * w) / patch_area


        x = F.normalize(x, p=2, dim=-3)  # 在通道维度归一化



        # [B, C, H, W] -> [B * C * n_h, p_h, n_w, p_w]
        x = x.reshape(b * c * num_patch_h, patch_size, num_patch_w, patch_size)
        # [B * C * n_h, p_h, n_w, p_w] -> [B * C * n_h, n_w, p_h, p_w]
        x = x.transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] -> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        x = x.reshape(b, c, num_patches, patch_area)

        i = torch.arange(patch_size)
        diagonal_indices = i * (patch_size + 1)
        x = x[:, :, :, diagonal_indices]


        # [B, C, N, P] -> [B, P, N, C]
        x = x.transpose(1, 3)
        # [B, P, N, C] -> [B*P, N, C]
        x = x.reshape(b * patch_size, num_patches, -1)

        x = torch.matmul(x, x.transpose(1, 2))
        x = x.reshape(b, patch_size, num_patches, num_patches)
        channel_avg = x.mean(dim=1, keepdim=True)  # 结果形状为 [B, 1, N, N]


        x = torch.cat([x, channel_avg], dim=1)


        x_in = x.view(b * (patch_size + 1), -1, num_patch_h, num_patch_w)
        rat_s = 0.05
        h_zero = num_patch_h
        w_zero = num_patch_w


        sigma = h_zero * rat_s, w_zero * rat_s
        b_z, c_z, h2_z, w2_z = x_in.shape
        key = str(x_in.shape) + str(rat_s)

        if key not in self.store:

            ind_r = torch.arange(h2_z, device=x_in.device).view(1, 1, h2_z, 1)
            ind_c = torch.arange(w2_z, device=x_in.device).view(1, 1, 1, w2_z)


            c_ind_r, c_ind_c = torch.meshgrid(
                torch.arange(h_zero, dtype=torch.float32, device=x_in.device),
                torch.arange(w_zero, dtype=torch.float32, device=x_in.device),
                indexing="ij"
            )
            cent_r = c_ind_r.reshape(1, c_z, 1, 1).to(x_in.device)
            cent_c = c_ind_c.reshape(1, c_z, 1, 1).to(x_in.device)


            def fn_gauss(x, u, s):
                return torch.exp(-(x - u) **2 / (2 * s** 2))

            gaus_r = fn_gauss(ind_r, cent_r, sigma[0])
            gaus_c = fn_gauss(ind_c, cent_c, sigma[1])
            out_g = 1 - gaus_r * gaus_c
            out_g = out_g.to(x_in.device)
            self.store[key] = out_g
        else:
            out_g = self.store[key]


        x_zeroed = out_g * x_in

        x = x_zeroed.reshape(b, (patch_size + 1), num_patch_h * num_patch_w, num_patch_h * num_patch_w)



        x = F.softmax(x * self.alpha, dim=-1) * F.softmax(x * self.alpha, dim=-2)
        new_x = x[:, -1, :, :].unsqueeze(1).expand(b, patch_area, num_patches, num_patches).clone()
        for index, diag_index in enumerate(diagonal_indices):
            new_x[:, diag_index, :, :] = x[:, index, :, :]
        x = new_x


        # [B, P, N, N] -> [B, N, N, P]
        x = x.transpose(1, 3)
        # [B, N, N, P] -> [B*N*n_h, n_w, p_h, p_w]
        x = x.reshape(b * num_patches * num_patch_h, num_patch_w, patch_size, patch_size)
        # [B*N*n_h, n_w, p_h, p_w] -> [B*N*n_h, p_h, n_w, p_w]
        x = x.transpose(1, 2)
        # [B*N*n_h, p_h, n_w, p_w] -> [B, N, H, W]
        x = x.reshape(b, num_patches, num_patch_h * patch_size, num_patch_w * patch_size)

        x = get_topk(x, k=self.topk, dim=-3)  # 在h1*w1维度选取top_k


        return x





class PASPP(nn.Module):
    def __init__(self, in_channel, depth=128, rate_dict=[2, 4, 8, 12]):
        super(PASPP, self).__init__()

        self.modules = []
        for index, n_rate in enumerate(rate_dict):
            self.modules.append(nn.Sequential(
                nn.Conv2d(in_channel, depth, 3, dilation=n_rate, padding=n_rate, bias=False),
                nn.BatchNorm2d(depth),
                nn.GELU(),
                nn.Conv2d(depth, int(depth / 4), 1, dilation=n_rate, bias=False),
            ))

        self.convs = nn.ModuleList(self.modules)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res[1]=res[0]+res[1]
        res[2]=res[1]+res[2]
        res[3]=res[2]+res[3]
        res[4]=res[3]+res[4]
        res[5]=res[4]+res[5]
        return torch.cat(res, 1)

def _upsample_like(src,tar):

    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src






class RWTN_Net(nn.Module):
    def __init__(self,Coefficient_3):
        super(RWTN_Net, self).__init__()
        self.Coefficient_3=Coefficient_3
        self.backbone = FRFE(Coefficient_3=Coefficient_3)
        #返回[b,40,160,160]
        self.ASFF2_40_80=ASFF2(level=1, multiplier=1, rfb=False, vis=False, dim=[80,40])
        #返回[b,160,20,20]
        self.ASFF2_160_160=ASFF2(level=0, multiplier=1, rfb=False, vis=False, dim=[160,160])
        #返回[b,80,80,80]
        self.ASFF3_40_80_160=ASFF3(level=1, multiplier=1, rfb=False, vis=False, dim=[160,80,40])
        #返回[b,160,40,40]
        self.ASFF3_80_160_160=ASFF3(level=1, multiplier=1, rfb=False, vis=False, dim=[160,160,80])

        self.head = ChannelAtt(c1=40,c2=80,c3=160,c4=160)
        self.corr1 = LASCC(topk=32)


        self.side1 = nn.Conv2d(32, 1, 3, padding=1)
        self.side2 = SConv_2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1,same=True)
        self.side3 = SConv_2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1,same=True)


        self.aspp = nn.Sequential(nn.LayerNorm([32,80,80]),
                                  PASPP(in_channel=32, depth=64, rate_dict=[2, 4, 8, 12,16,20]),
                                  nn.Conv2d(96, 64, 1),
                                  nn.BatchNorm2d(64),
                                  nn.GELU(),
                                  )
        self.rsu = AUB6(96,64)

        self.outconv = nn.Conv2d(4, 1, 1)


    def forward(self, x):
        x0 = x
        #FRFE
        x1, x2, x3, x4 = self.backbone(x)
        #AMAF
        x11=x1
        x22=x2
        x33=x3
        x44=x4
        x1=self.ASFF2_40_80([x11,x22])
        x2=self.ASFF3_40_80_160([x11,x22,x33])
        x3=self.ASFF3_80_160_160([x22,x33,x44])
        x4=self.ASFF2_160_160([x33,x44])
        d0 = self.head(x1, x2, x3, x4)

        #LASCC
        d0 = self.corr1(d0)
        d1=d0
        #PASPP
        s1 = self.aspp(d0)

        y=torch.cat([d1,s1],dim=1)

        #AUB6
        s2 = self.rsu(y)

        d2 = _upsample_like(self.side2(s1,self.Coefficient_3), x0)

        d3 = _upsample_like(self.side3(s2,self.Coefficient_3), x0)


        return d3,d2
