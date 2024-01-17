import torch
import torch.nn.functional as F

def softArgMax(x, dim, alpha=1):
    return F.softmax(x*alpha, dim)
    
def softMax(x, dim, alpha=1):
    # return 1./alpha * torch.log(torch.exp(alpha * x).sum(dim))
    return (softArgMax(x, dim, alpha) * x).sum(dim, )

def softAbs(x, alpha=1):
    return softMax(torch.cat([x.unsqueeze(0), -x.unsqueeze(0)], dim=0), 0, alpha.unsqueeze(0))

def softArgMin(x, dim, alpha=1):
    return softArgMax(-x, dim, alpha) # F.softmax computes softArgMax

def softMax(x, dim, alpha=1, keepdims=False):
    in_shape = list(x.shape)
    
    softargmax = softArgMax(x, dim, alpha)
    softmax = (x * softargmax).sum(dim)
    if keepdims:
        out_shape = in_shape
        out_shape[dim] = 1
        softmax = softmax.view(*out_shape)
    return softmax

def softMin(x, dim, alpha=1, keepdims=False):
    in_shape = list(x.shape)
    
    softargmin = softArgMin(x, dim, alpha)
    softmin = (x * softargmin).sum(dim)
    if keepdims:
        out_shape = in_shape
        out_shape[dim] = 1
        softmin = softmin.view(*out_shape)
    return softmin
def expand(x, dim):
    in_size = x.shape
    len_cand = in_size[dim]
    # out = x.expand([len_cand,] + list(in_size))
    out = torch.cat([x.unsqueeze(0) for _ in range(len_cand)], 0)
    return out

def softArgMedian(x, dim, alpha=1, beta=1):

    x1 = expand(x, dim).transpose(0, dim+1)
    x2 = expand(x, dim)
    tmp = softAbs(beta * x1 - x2, alpha.unsqueeze(0)).mean(0)
    tmp = softArgMin(tmp, dim, alpha)

    return tmp

def softMedian(x, dim, alpha=1, beta=1, keepdims=False):
    in_shape = list(x.shape)
    
    softargmedian = softArgMedian(x, dim, alpha, beta)
    softmedian = (x * softargmedian).sum(dim)
    if keepdims:
        out_shape = in_shape
        out_shape[dim] = 1
        softmedian = softmedian.view(*out_shape)
    return softmedian

if __name__ == '__main__':
    a = torch.randn([7,5])
    int0 = torch.tensor([0])
    int1 = torch.tensor([1])
    intinf = torch.tensor([1000])
    median = a.median(0)[0]
    m1 = softMedian(a, 0, alpha=intinf, beta=int1)
    m2 = softMedian(a, 0, alpha=int1, beta=int1)
    m3 = softMedian(a, 0, alpha=int0, beta=int0)
    mean = a.mean(0)
    print(a)

    print(median)
    
    print("When alpha is large, softmedian==median", m1)
    print(m2)
    print("When alpha is small, softmedian==mean", m3)
    print(mean)

    print(a.max(0)[0], a.min(0)[0])
    print(softMedian(a, 0, alpha=intinf, beta=intinf))
    print(softMedian(a, 0, alpha=intinf, beta=int0))
    #print(softMedian(a, 0, alpha=int0, beta=intinf))
