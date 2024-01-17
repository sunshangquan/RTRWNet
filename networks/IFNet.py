import torch
import torch.nn as nn
import torch.nn.functional as F
from warplayer import warp
from refine import *

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.PReLU(out_planes)
    )

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )

class IFBlock(nn.Module):
    def __init__(self, in_planes, c=32, seq=7):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
            )
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        self.lastconv = nn.ConvTranspose2d(c*seq, 2*(seq-1), 4, 2, 1)

    def forward(self, x, flow, scale):
        if scale != 1:
            x = F.interpolate(x, scale_factor = 1. / scale, mode="bilinear", align_corners=False)
        if flow != None:
            flow = F.interpolate(flow, scale_factor = 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            x = torch.cat((x, flow), 1)
            print(flow.shape)
        print(x.shape)
        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = self.lastconv(x)
        tmp = F.interpolate(tmp, scale_factor = scale * 2, mode="bilinear", align_corners=False)
        flow = tmp * scale * 2
        return flow
    
class IFNet(nn.Module):
    def __init__(self, seq=7):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(3, c=240, seq=seq)
        self.block1 = IFBlock(6, c=150)
        self.block2 = IFBlock(13+4, c=90)
        self.block_tea = IFBlock(16+4, c=90)
        self.contextnet = Contextnet()
        self.unet = Unet()

    def forward(self, x, scale=[4,2,1], timestep=0.5):
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        img2 = x[:, 6:9]
        img3 = x[:, 9:12]
        img4 = x[:, 12:15]
        img5 = x[:, 15:18]
        img6 = x[:, 18:21]
        gt = x[:, 21:] # None
        flow_list = []
        merged = []
        warped_img0 = img0
        warped_img1 = img1
        warped_img2 = img2
        warped_img4 = img4
        warped_img5 = img5
        warped_img6 = img6
        warped_imgs = (warped_img0, warped_img1, warped_img2, warped_img4, warped_img5, warped_img6)
        flow = None 
        loss_distill = 0
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow != None:
                print(warped_imgs)
                flow_d = stu[i](torch.cat((x,)+warped_imgs, 1), flow, scale=scale[i])
                flow = flow + flow_d
            else:
                flow = stu[i](x, None, scale=scale[i])
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            warped_img2 = warp(img2, flow[:, 4:6])
            warped_img4 = warp(img4, flow[:, 6:8])
            warped_img5 = warp(img5, flow[:, 8:10])
            warped_img6 = warp(img6, flow[:, 10:12])
            warped_imgs = (warped_img0, warped_img1, warped_img2, warped_img4, warped_img5, warped_img6)
            merged.append(warped_imgs)
        
        for i in range(3):
            merged[i] = merged[i][0] + merged[i][1] + merged[i][2] + merged[i][4] + merged[i][5] + merged[i][6] 
        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        c2 = self.contextnet(img2, flow[:, 4:6])
        c4 = self.contextnet(img4, flow[:, 6:8])
        c5 = self.contextnet(img5, flow[:, 8:10])
        c6 = self.contextnet(img6, flow[:, 10:12])
        cs = (c0, c1, c2, c4, c5, c6)
        tmp = self.unet(x, warped_imgs, (flow), cs)
        res = tmp[:, :3] * 2 - 1
        merged[2] = torch.clamp(merged[2] + res, 0, 1)
        return merged, flow_list

if __name__ == "__main__":
    x = torch.randn([2,21,128,256]).cuda()
    net = IFNet().cuda()
    merged, flow = net(x)
    print(merged[-1].shape)
