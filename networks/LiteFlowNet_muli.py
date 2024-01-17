import torch
from networks.correlation_package.correlation import Correlation
import math

backwarp_tenGrid={}

def backwarp(tenInput, tenFlow, ):
    b = tenFlow.shape[0]
    grid_key = str(tenFlow.shape)+",device:"+str(tenFlow.device)
    if grid_key not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, -1).expand(b, -1, tenFlow.shape[2], -1)
        tenVer = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, -1, 1).expand(b, -1, -1, tenFlow.shape[3])

        backwarp_tenGrid[grid_key] = torch.cat([ tenHor, tenVer ], 1).to(tenFlow)
        print("create backwarp_tenGrid", grid_key)
    # end

    tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)
    g = (backwarp_tenGrid[grid_key] + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)

class LightFlowNet(torch.nn.Module):
    def __init__(self):
        super(LightFlowNet, self).__init__()

        class Features(torch.nn.Module):
            def __init__(self):
                super(Features, self).__init__()
                self.sq = 7
                self.single_channels = [12, 12, 24, 36, 48, 60]
                self.channels = [i*self.sq for i in self.single_channels]
                chs = self.channels

                self.netOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=21, out_channels=chs[0], kernel_size=7, stride=1, padding=3),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=chs[0], out_channels=chs[1], kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=chs[1], out_channels=chs[1], kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=chs[1], out_channels=chs[1], kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=chs[1], out_channels=chs[2], kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=chs[2], out_channels=chs[2], kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=chs[2], out_channels=chs[3], kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=chs[3], out_channels=chs[3], kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=chs[3], out_channels=chs[4], kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netSix = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=chs[4], out_channels=chs[5], kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
            # end

            def forward(self, tenInput):
                tenOne = self.netOne(tenInput)
                tenTwo = self.netTwo(tenOne)
                tenThr = self.netThr(tenTwo)
                tenFou = self.netFou(tenThr)
                tenFiv = self.netFiv(tenFou)
                tenSix = self.netSix(tenFiv)

                sin_chs = self.single_channels
                cen = self.sq // 2
                ind_cen = [list(range(cen*i, (cen+1)*i)) for i in sin_chs]
                ind_nei = [list(range(0, cen*i)) + list(range((cen+1)*i, self.sq*i)) for i in sin_chs]
                tenFeaturesFirst, tenFeaturesSecond = [ tenOne[:,ind_cen[0]], tenTwo[:,ind_cen[1]], tenThr[:,ind_cen[2]], tenFou[:,ind_cen[3]], tenFiv[:,ind_cen[4]], tenSix[:,ind_cen[5]] ], \
                [ tenOne[:,ind_nei[0]], tenTwo[:,ind_nei[1]], tenThr[:,ind_nei[2]], tenFou[:,ind_nei[3]], tenFiv[:,ind_nei[4]], tenSix[:,ind_nei[5]] ]
                # tenFeaturesFirst = [torch.cat([fea]*(self.sq-1), 1) for fea in tenFeaturesFirst]
                return tenFeaturesFirst, tenFeaturesSecond,
            # end
        # end

        class Matching(torch.nn.Module):
            def __init__(self, intLevel):
                super(Matching, self).__init__()

                self.fltBackwarp = [ 0.0, 0.0, 0.0, 5.0, 2.5, 1.25, 0.625 ][intLevel]

                
                self.crossCorr = Correlation(pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1)
                

                if intLevel == 4:
                    self.autoCorr = Correlation(pad_size=6, kernel_size=1, max_displacement=6, stride1=1, stride2=2)
                elif intLevel == 3:
                    self.autoCorr = Correlation(pad_size=8, kernel_size=1, max_displacement=8, stride1=1, stride2=2)

                if intLevel > 4:
                    self.confFeat = None
                    self.corrFeat = None
                
                if intLevel <= 4:
                    self.confFeat = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=[0, 0, 0, 6+81, 6+49][intLevel], out_channels=128, kernel_size=3, stride=1, padding=1),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                    )
                    self.dispNet = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32, out_channels=12, kernel_size=5, stride=1, padding=2)
                        )
                    self.confNet = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32, out_channels=6, kernel_size=5, stride=1, padding=2),
                        torch.nn.Sigmoid()
                        )

                    self.corrFeat = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=[0, 0, 0, 24+81+6, 36+81+6][intLevel], out_channels=128, kernel_size=3, stride=1, padding=1),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                    self.corrScalar = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        torch.nn.Conv2d(in_channels=32, out_channels=81, kernel_size=1, stride=1, padding=0)
                    )
                    self.corrOffset = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        torch.nn.Conv2d(in_channels=32, out_channels=81, kernel_size=1, stride=1, padding=0)
                    )

                # end

                if intLevel == 6:
                    self.netUpflow = None

                elif intLevel != 6:
                    self.netUpflow = torch.nn.ConvTranspose2d(in_channels=12, out_channels=12, kernel_size=4, stride=2, padding=1, bias=False, groups=2)

                if intLevel == 4 or intLevel == 3:
                    self.netUpconf = torch.nn.ConvTranspose2d(in_channels=6, out_channels=6, kernel_size=4, stride=2, padding=1, bias=False, groups=1)
                # end

                self.netMain = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=81, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=12, kernel_size=[ 0, 0, 0, 5, 5, 3, 3 ][intLevel], stride=1, padding=[ 0, 0, 0, 2, 2, 1, 1 ][intLevel])
                )
            # end

            def forward(self, tenFeaturesFirst, tenFeaturesSecond, tenFlow, tenConf):
                if self.confFeat:
                    tenConf = self.netUpconf(tenConf)
                    tenCorrelation = torch.nn.functional.leaky_relu(input=self.autoCorr(tenFeaturesFirst, tenFeaturesFirst), negative_slope=0.1, inplace=False)
                    confFeat = self.confFeat(torch.cat([tenCorrelation, tenConf], 1))
                    tenConf = self.confNet(confFeat)
                    tenDisp = self.dispNet(confFeat)
                    
                if tenFlow is not None:
                    tenFlow = self.netUpflow(tenFlow)
                # end
                if self.corrFeat:
                    tmp_list = []
                    for i in range(6):
                        flow = tenFlow[:,i*2:(i+1)*2]
                        disp = tenDisp[:,i*2:(i+1)*2]
                        tmp = backwarp(tenInput=flow, tenFlow=disp)
                        tmp_list.append(tmp)
                    tenFlow = torch.cat(tmp_list, 1)

                if tenFlow is not None:
                    tmp_list = []
                    ch_f = tenFeaturesSecond.shape[1]
                    sing_ch = ch_f // 6
                    for i in range(6):
                        feat = tenFeaturesSecond[:,i*sing_ch:(i+1)*sing_ch]
                        flow = tenFlow[:,i*2:(i+1)*2]
                        tmp = backwarp(tenInput=feat, tenFlow=flow * self.fltBackwarp)
                        tmp_list.append(tmp)
                    tenFeaturesSecond = torch.cat(tmp_list, 1)
                # end
                tenFeaturesFirsts = torch.cat([tenFeaturesFirst]*6, 1)
                tenCorrelation = torch.nn.functional.leaky_relu(input=self.crossCorr(tenFeaturesFirsts, tenFeaturesSecond), negative_slope=0.1, inplace=False)
                

                if self.corrFeat:
                    corrfeat = self.corrFeat(torch.cat([tenFeaturesFirst, tenCorrelation, tenConf], 1))
                    corrscalar = self.corrScalar(corrfeat)
                    corroffset = self.corrOffset(corrfeat)
                    tenCorrelation = corrscalar * tenCorrelation + corroffset
                
                return (tenFlow if tenFlow is not None else 0.0) + self.netMain(tenCorrelation), tenConf
            # end
        # end

        class Subpixel(torch.nn.Module):
            def __init__(self, intLevel):
                super(Subpixel, self).__init__()

                self.fltBackward = [ 0.0, 0.0, 0.0, 5.0, 2.5, 1.25, 0.625 ][intLevel]

                self.netMain = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=[ 0, 0, 0, 24*7+12, 36*7+12, 48*7+12, 60*7+12 ][intLevel], out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=12, kernel_size=[ 0, 0, 0, 5, 5, 3, 3 ][intLevel], stride=1, padding=[ 0, 0, 0, 2, 2, 1, 1 ][intLevel])
                )
            # end

            def forward(self, tenFeaturesFirst, tenFeaturesSecond, tenFlow):
                if tenFlow is not None:
                    tmp_list = []
                    ch_f = tenFeaturesSecond.shape[1]
                    sing_ch = ch_f // 6
                    for i in range(6):
                        feat = tenFeaturesSecond[:,i*sing_ch:(i+1)*sing_ch]
                        flow = tenFlow[:,i*2:(i+1)*2]
                        tmp = backwarp(tenInput=feat, tenFlow=flow * self.fltBackward)
                        tmp_list.append(tmp)
                    tenFeaturesSecond = torch.cat(tmp_list, 1)
                # end

                return (tenFlow if tenFlow is not None else 0.0) + self.netMain(torch.cat([ tenFeaturesFirst, tenFeaturesSecond, tenFlow ], 1))
            # end
        # end

        class Regularization(torch.nn.Module):
            def __init__(self, intLevel):
                super(Regularization, self).__init__()

                self.fltBackward = [ 0.0, 0.0, 0.0, 5.0, 2.5, 1.25, 0.625 ][intLevel]

                self.intUnfold = [ 0, 0, 7, 5, 5, 3, 3 ][intLevel]

                if intLevel > 4:
                    self.netFeat = torch.nn.Sequential()

                elif intLevel <= 4:
                    self.netFeat = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=[ 0, 0, 12, 24, 36, 48, 60 ][intLevel], out_channels=128, kernel_size=1, stride=1, padding=0),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                    )

                # end

                self.netMain = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=[ 0, 0, 128+12+6, 128+12+6, 128+12+6, 48+12+6, 60+12+6 ][intLevel], out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                

                if intLevel >= 5:
                    self.netDist = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32, out_channels=[ 0, 0, 49, 25*6, 25*6, 9*6, 9*6 ][intLevel], kernel_size=[ 0, 0, 7, 5, 5, 3, 3 ][intLevel], stride=1, padding=[ 0, 0, 3, 2, 2, 1, 1 ][intLevel])
                    )

                elif intLevel < 5:
                    self.netDist = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32, out_channels=[ 0, 0, 0, 25*6, 25*6, 9*6, 9*6 ][intLevel], kernel_size=([ 0, 0, 0, 5, 5, 3, 3 ][intLevel], 1), stride=1, padding=([ 0, 0, 0, 2, 2, 1, 1 ][intLevel], 0)),
                        torch.nn.Conv2d(in_channels=[ 0, 0, 0, 25*6, 25*6, 9*6, 9*6 ][intLevel], out_channels=[ 0, 0, 0, 25*6, 25*6, 9*6, 9*6 ][intLevel], kernel_size=(1, [ 0, 0, 0, 5, 5, 3, 3 ][intLevel]), stride=1, padding=(0, [ 0, 0, 0, 2, 2, 1, 1 ][intLevel]))
                    )
                
                if intLevel == 5 or intLevel == 4:
                    self.confNet = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32, out_channels=6, kernel_size=[0, 0, 0, 0, 5, 3][intLevel], stride=1, padding=[0, 0, 0, 0, 2, 1][intLevel]),
                        torch.nn.Sigmoid()
                    )
                else:
                    self.confNet = None
                # end

                self.netScaleX = torch.nn.Conv2d(in_channels=[ 0, 0, 49, 25*6, 25*6, 9*6, 9*6 ][intLevel], out_channels=6, kernel_size=1, stride=1, padding=0)
                self.netScaleY = torch.nn.Conv2d(in_channels=[ 0, 0, 49, 25*6, 25*6, 9*6, 9*6 ][intLevel], out_channels=6, kernel_size=1, stride=1, padding=0)
            # eny

            def forward(self, tenInput, tenFeaturesFirst, tenFeaturesSecond, tenFlow):
                tenFirst = tenInput[:, 9:12]
                tenSecond = tenInput[:, list(range(9))+list(range(12,21))]
                tmp_list = []
                tenDifference = 0
                for i in range(6):
                    frame = tenSecond[:,i*3:(i+1)*3]
                    flow = tenFlow[:,i*2:(i+1)*2]
                    warpSecond = backwarp(tenInput=frame, tenFlow=flow * self.fltBackward)
                    tmp = (tenFirst - warpSecond).pow(2.0).sum(1, True).sqrt().detach()
                    tmp_list.append(tmp)
                tenDifference = torch.cat(tmp_list, 1)

                tenFeaturesFirst = self.netFeat(tenFeaturesFirst)

                mainfeat = self.netMain(torch.cat([ tenDifference, tenFlow - tenFlow.view(tenFlow.shape[0], 12, -1).mean(2, True).view(tenFlow.shape[0], 12, 1, 1), tenFeaturesFirst ], 1))
                tenDist = self.netDist(mainfeat)
                
                tenConf = None
                if self.confNet:
                    tenConf = self.confNet(mainfeat)

                tenDist = tenDist.pow(2.0).neg()
                tenDist = (tenDist - tenDist.max(1, True)[0]).exp()

                bn, _, h, w = tenDist.shape
                tenDivisor = tenDist.view(bn, 6, -1, h, w).sum(2).reciprocal()

                tenScaleX = self.netScaleX(tenDist * torch.nn.functional.unfold(input=tenFlow[:, 0:12:2, :, :], kernel_size=self.intUnfold, stride=1, padding=int((self.intUnfold - 1) / 2)).view_as(tenDist)) * tenDivisor
                tenScaleY = self.netScaleY(tenDist * torch.nn.functional.unfold(input=tenFlow[:, 1:12:2, :, :], kernel_size=self.intUnfold, stride=1, padding=int((self.intUnfold - 1) / 2)).view_as(tenDist)) * tenDivisor

                return torch.cat([ tenScaleX.unsqueeze(2), tenScaleY.unsqueeze(2) ], 2).view(bn, -1, h, w), tenConf
            # end
        # end

        self.netFeatures = Features()
        self.netMatching = torch.nn.ModuleList([ Matching(intLevel) for intLevel in [ 3, 4, 5, 6 ] ])
        self.netSubpixel = torch.nn.ModuleList([ Subpixel(intLevel) for intLevel in [ 3, 4, 5, 6 ] ])
        self.netRegularization = torch.nn.ModuleList([ Regularization(intLevel) for intLevel in [ 3, 4, 5, 6 ] ])
        # self.load_state_dict(torch.load('network-sintel.pytorch'))
    # end

    def forward(self, tenInput):
        shape = tenInput.shape
        h, w = shape[-2:]
        intPreWidth = int(math.floor(math.ceil(w / 32.0) * 32.0))
        intPreHeight = int(math.floor(math.ceil(h / 32.0) * 32.0))
        if len(shape) == 5:
            tenInput = tenInput - torch.mean(tenInput, (3, 4), keepdim=True)
            tenInput = tenInput.view(b, -1, h, w)
        else:
            tenInput = tenInput - torch.mean(tenInput, (2, 3), keepdim=True)

        tenFeaturesFirst, tenFeaturesSecond = self.netFeatures(tenInput)
        
        tenInput = [ tenInput ]
        
        for intLevel in [ 2, 3, 4, 5 ]:
            tenInput.append(torch.nn.functional.interpolate(input=tenInput[-1], size=(tenFeaturesFirst[intLevel].shape[2], tenFeaturesFirst[intLevel].shape[3]), mode='bilinear', align_corners=False))
        # end

        tenFlow = None
        tenConf = None

        for intLevel in [ -1, -2, -3, -4 ]:
            tenFlow, tenConf = self.netMatching[intLevel](tenFeaturesFirst[intLevel], tenFeaturesSecond[intLevel], tenFlow, tenConf)
            tenFlow = self.netSubpixel[intLevel](tenFeaturesFirst[intLevel], tenFeaturesSecond[intLevel], tenFlow)
            tenFlow, tenConf = self.netRegularization[intLevel](tenInput[intLevel], tenFeaturesFirst[intLevel], tenFeaturesSecond[intLevel], tenFlow)
        # end

        tenFlow *= 20.0

        tenFlow = torch.nn.functional.interpolate(
            input=tenFlow, 
            size=(h, w), 
            mode='bilinear', 
            align_corners=False
        )
        tenFlow[:, 0:12:2, :, :] *= float(w) / float(intPreWidth)
        tenFlow[:, 1:12:2, :, :] *= float(h) / float(intPreHeight)
        return tenFlow
    # end
# end

if __name__ == "__main__":
    netNetwork = LightFlowNet().cuda().eval()
    a = torch.randn(12,21, 128,128).cuda()
    b  = netNetwork(a)
    print(b.shape)
