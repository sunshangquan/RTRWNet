from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange

from networks.SoftMedian import softMedian


backwarp_tenGrid={}
def flow_warping(tenInput, tenFlow, ):
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


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class ChannelNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(ChannelNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class Norm(nn.Module):
    def __init__(self, dim, Norm_type):
        super(Norm, self).__init__()
        if Norm_type == 'Group':
#            self.body = BiasFree_LayerNorm(dim)
            pass
        elif Norm_type == 'Channel':
            self.body = ChannelNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

#        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2, = self.dwconv(x).chunk(2, dim=1)
#        x = F.gelu(x1) * x2
        x = F.sigmoid(2 * x1) * x1 * x2
#        x = F.gelu(x3) + x

        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Sequential(
                                         nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias),
                                         nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
                                         #nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
                           )
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, Norm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = Norm(dim, Norm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = Norm(dim, Norm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
#        x = self.norm1(x + self.attn(x))
#        x = self.norm2(x + self.ffn(x))
        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
class Backbone(nn.Module):
    def __init__(self, 
        inp_channels=21, 
        out_channels=3, 
        dim = 48,
        num_blocks = [1,1,1,1],  # 1,2,2,4
        num_refinement_blocks = 2,
        heads = [1,2,4,8],
        ffn_expansion_factor = 8./3.,
        bias = False,
        Norm_type = 'Channel',   ## Other options 'Group', 'Instance', 'Layer'
    ):

        super(Backbone, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, Norm_type=Norm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, Norm_type=Norm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, Norm_type=Norm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, Norm_type=Norm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, Norm_type=Norm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, Norm_type=Norm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
#        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**0), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, Norm_type=Norm_type) for i in range(num_blocks[0])])
#        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**0), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, Norm_type=Norm_type) for i in range(num_refinement_blocks)])
#        self.output = nn.Conv2d(int(dim*2**0), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, Norm_type=Norm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, Norm_type=Norm_type) for i in range(num_refinement_blocks)])
        
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
      

        self.dim = dim
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 0)
#                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        '''
    def forward(self, frame):
        inp_img = frame
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
#        inp_dec_level3 = inp_dec_level3 + out_enc_level3
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
#        inp_dec_level2 = inp_dec_level2 + out_enc_level2
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
#        inp_dec_level1 = inp_dec_level1 + out_enc_level1
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1)


        return out_dec_level1

class Model(nn.Module):
    def __init__(self, opts, scales=[1]):
        super(Model, self).__init__()
        self.scales = scales
        self.ifAlign = opts.ifAlign
        self.ifAggregate = opts.ifAggregate
        self.frame_restorer = Backbone(inp_channels=39, out_channels=3, num_blocks=[1,1,1,1], num_refinement_blocks=2)
        if self.ifAlign:
            self.flow_estimator = Backbone(inp_channels=21, out_channels=12, num_blocks=[1,1,1,1], num_refinement_blocks=2)
            if self.ifAggregate:
                self.frame_restorer = Backbone(inp_channels=39, out_channels=23, num_blocks=[1,1,1,1], num_refinement_blocks=2)
        else:
            self.flow_estimator = nn.Sequential()


    def predict_flow(self, input_, scales):
        flow_list = []
        for j, scale in enumerate(scales):
            input_scaled = F.interpolate(input_, scale_factor=1./scales[j], mode="bilinear", align_corners=False)
            flow_inward = self.flow_estimator(input_scaled)
            flow_list.append(flow_inward)
        return flow_list

    def forward(self, inputs, ifInferece=True, flow_tea=None):
        bn, sq, ch_f, h, w = inputs.shape
        flow_list = []
        frame_center = inputs[:,sq//2]
        if self.ifAlign:
            seq_around = inputs[:,[i for i in range(sq) if i != sq//2]]

            input_ = inputs.contiguous().view(bn, -1, h, w)
            
            if flow_tea is None:
                flow_inward = self.flow_estimator(input_)
                flow_list.append(flow_inward)
                flow_inward = flow_list[-1]
            else:
                flow_inward = flow_tea
            flow_inward = flow_inward.contiguous().view(bn, sq-1, 2, h, w)
            warp_inward = [flow_warping(seq_around[:,i], flow_inward[:,i]) for i in range(sq-1)]

            seq_input = torch.cat(
                                  [seq_around[:,i] for i in range(3)]+
                                  [frame for frame in warp_inward[:3]]
                                   +[frame_center]+
                                  [frame for frame in warp_inward[3:]]
                                  +[seq_around[:,3+i] for i in range(3)]
                                  ,1).detach()

        else:
            seq_input = inputs.contiguous().view(bn, -1, h, w)

        
        out = self.frame_restorer(seq_input)
        if self.ifAlign:
            if self.ifAggregate:
#                alpha = torch.ones_like(out)
                alpha = out[:,-2:-1].unsqueeze(2)
                beta = out[:,-1:].unsqueeze(2)
                out = out[:,:-2].view(bn, sq, ch_f, h, w)
                out = softMedian(out, dim=1, alpha=alpha, beta=beta)
        
        return out + frame_center, flow_list, seq_input


