
# spatial and temporal feature fusion for change detection of remote sensing images
# STNet11
# Author: xwma
# Time: 2022.11.2
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from rscd.models.decoderheads.CBAM import *

sys.path.append('rscd')


def conv_3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )


def dsconv_3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel),
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, groups=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )


def conv_1x1(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SelfAttentionBlock(nn.Module):
    """
    query_feats: (B, C, h, w)
    key_feats: (B, C, h, w)
    value_feats: (B, C, h, w)

    output: (B, C, h, w)
    """

    def __init__(self, key_in_channels, query_in_channels, transform_channels, out_channels,
                 key_query_num_convs, value_out_num_convs):
        super(SelfAttentionBlock, self).__init__()
        self.key_project = self.buildproject(
            in_channels=key_in_channels,
            out_channels=transform_channels,
            num_convs=key_query_num_convs,
        )
        self.query_project = self.buildproject(
            in_channels=query_in_channels,
            out_channels=transform_channels,
            num_convs=key_query_num_convs
        )
        self.value_project = self.buildproject(
            in_channels=key_in_channels,
            out_channels=transform_channels,
            num_convs=value_out_num_convs
        )
        self.out_project = self.buildproject(
            in_channels=transform_channels,
            out_channels=out_channels,
            num_convs=value_out_num_convs
        )
        self.transform_channels = transform_channels

    def forward(self, query_feats, key_feats, value_feats):
        batch_size = query_feats.size(0)

        query = self.query_project(query_feats)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()  # (B, h*w, C)

        key = self.key_project(key_feats)
        key = key.reshape(*key.shape[:2], -1)  # (B, C, h*w)

        value = self.value_project(value_feats)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous()  # (B, h*w, C)

        sim_map = torch.matmul(query, key)

        sim_map = (self.transform_channels ** -0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)  # (B, h*w, K)

        context = torch.matmul(sim_map, value)  # (B, h*w, C)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *query_feats.shape[2:])  # (B, C, h, w)

        context = self.out_project(context)  # (B, C, h, w)
        return context

    def buildproject(self, in_channels, out_channels, num_convs):
        convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        for _ in range(num_convs - 1):
            convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        if len(convs) > 1:
            return nn.Sequential(*convs)
        return convs[0]

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)


        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        a_h = a_h.expand(-1,-1,h,w)
        a_w = a_w.expand(-1, -1, h, w)

        # out = identity * a_w * a_h

        return a_w , a_h


class DGF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DGF, self).__init__()
        self.catconvA = dsconv_3x3(in_channel * 2, in_channel)
        self.catconvB = dsconv_3x3(in_channel * 2 , in_channel)
        self.catconv = dsconv_3x3(in_channel * 2, out_channel)
        self.convA = nn.Conv2d(in_channel, 1, 1)
        self.convB = nn.Conv2d(in_channel, 1, 1)
        self.sigmoid = nn.Sigmoid()
        #self.in_channels = in_channel
        #self.out_channels = out_channel



        # 注意力机制更新：也要适配 out_channels
        self.coAtt_1 = CoordAtt(inp=in_channel, oup=in_channel, reduction=16)

    def forward(self, xA, xB):
        #x_A=self.catconvA(xA)
        #A_weight = self.sigmoid(self.convA(x_A))
        #xA = A_weight * xA

        x_diff = xA - xB




        #CoordAtt
        d_aw, d_ah = self.coAtt_1(x_diff)
        out = x_diff * d_aw * d_ah

        x_diffA = self.catconvA(torch.cat([out, xA], dim=1))#通道维度的连接
        x_diffB = self.catconvB(torch.cat([out, xB], dim=1))


        A_weight = self.sigmoid(self.convA(x_diffA))
        B_weight = self.sigmoid(self.convB(x_diffB))

        xA = A_weight * xA
        xB = B_weight * xB



        x = self.catconv(torch.cat([xA, xB], dim=1))


        return x




class CFF(nn.Module):
    def __init__(self, in_channel):
        super(CFF, self).__init__()
        self.conv_small = conv_1x1(in_channel, in_channel)
        self.conv_big = conv_1x1(in_channel, in_channel)
        self.catconv = conv_3x3(in_channel *2 , in_channel)

        self.ca = ChannelAttention(in_channel)


        self.attention = SelfAttentionBlock(
            key_in_channels=in_channel,
            query_in_channels=in_channel,
            transform_channels=in_channel // 2,
            out_channels=in_channel,
            key_query_num_convs=2,
            value_out_num_convs=1
        )



    def forward(self, x_small, x_big):
        img_size = x_big.size(2), x_big.size(3)
        x_small = F.interpolate(x_small, img_size, mode="bilinear", align_corners=False)
        x = self.conv_small(x_small) + self.conv_big(x_big)



        new_x = self.attention(x, x, x_big)

        out = self.catconv(torch.cat([new_x, x_big], dim=1))


        out_last = self.ca(out) * out


        return out_last


class SSFF(nn.Module):
    def __init__(self):
        super(SSFF, self).__init__()
        self.spatial = SpatialAttention()

    def forward(self, x_small, x_big):
        img_shape = x_small.size(2), x_small.size(3)
        big_weight = self.spatial(x_big)
        big_weight = F.interpolate(big_weight, img_shape, mode="bilinear", align_corners=False)
        x_small = big_weight * x_small
        return x_small








class LightDecoder(nn.Module):
    def __init__(self, in_channel, num_class, layer_num):
        super(LightDecoder, self).__init__()
        self.layer_num = layer_num
        #self.channel_attention = ChannelAttention(in_channel * layer_num)
        self.catconv = conv_3x3(in_channel * layer_num, in_channel)

        self.decoder = nn.Conv2d(in_channel, num_class, 1)#1x1分类器

        #self.brm = BoundaryRefine(in_channel)


    def forward(self, x1, x2, x3,x4):


        x2 = F.interpolate(x2, scale_factor=2, mode="bilinear")
        x3 = F.interpolate(x3, scale_factor=4, mode="bilinear")
        x4 = F.interpolate(x4, scale_factor=8, mode="bilinear") if self.layer_num == 4 else None



        x = torch.cat([x1, x2, x3,x4], dim=1) if self.layer_num == 4 else torch.cat([x1, x2, x3], dim=1)

        x = self.catconv(x) # 降维


        out = self.decoder(x)
        return out

class DGCFFNet(nn.Module):
    def __init__(self, num_class, channel_list, transform_feat, layer_num):
        super(DGCFFNet, self).__init__()

        self.layer_num = layer_num

        self.dgf1 = DGF(channel_list[0], transform_feat)
        self.dgf2 = DGF(channel_list[1], transform_feat)
        self.dgf3 = DGF(channel_list[2], transform_feat)
        self.dgf4 = DGF(channel_list[3], transform_feat)

        self.cff1 = CFF(transform_feat)
        self.cff2 = CFF(transform_feat)
        self.cff3 = CFF(transform_feat)


        self.lightdecoder = LightDecoder(transform_feat, num_class, layer_num)


    def forward(self, x):

        featuresA, featuresB = x
        xA1, xA2, xA3, xA4 = featuresA
        xB1, xB2, xB3, xB4 = featuresB



        x1 = self.dgf1(xA1, xB1)
        x2 = self.dgf2(xA2, xB2)
        x3 = self.dgf3(xA3, xB3)
        x4 = self.dgf4(xA4, xB4) if self.layer_num == 4 else None

        xlast = x4 if self.layer_num == 4 else x3


        x1_new = self.cff1(x2, x1)

        x2_new = self.cff2(x3, x2)

        x3_new = self.cff3(x4, x3) if self.layer_num == 4 else x3


        out = self.lightdecoder(x1_new, x2_new, x3_new,xlast)

        out = F.interpolate(out, scale_factor=4, mode="bilinear")

        return out


