"""
Orthogonal Weight Normalization: Solution to Optimization over Multiple Dependent Stiefel Manifolds in Deep Neural Networks
AAAI 2018

Authors: Lei Huang
"""
import torch.nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable
from typing import List
from torch.autograd.function import once_differentiable

__all__ = ['OWN_Conv2d']

#  norm funcitons--------------------------------


class IdentityModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(IdentityModule, self).__init__()

    def forward(self, input: torch.Tensor):
        return input

class OWNNorm(torch.nn.Module):
    def __init__(self, norm_groups=1, *args, **kwargs):
        super(OWNNorm, self).__init__()
        self.norm_groups = norm_groups

    def matrix_power3(self, Input):
        B=torch.bmm(Input, Input)
        return torch.bmm(B, Input)

    def forward(self, weight: torch.Tensor):
        assert weight.shape[0] % self.norm_groups == 0
        Z = weight.view(self.norm_groups, weight.shape[0] // self.norm_groups, -1)  # type: torch.Tensor
        Zc = Z - Z.mean(dim=-1, keepdim=True)
        S = torch.matmul(Zc, Zc.transpose(1, 2))
        wm = torch.randn(S.shape).to(S)
        for i in range(self.norm_groups):
            U, Eig, _ = S[i].svd()
            Scales = Eig.rsqrt().diag()
            wm[i] = U.mm(Scales).mm(U.t())
        W = wm.matmul(Zc)
        return W.view_as(weight)

    def extra_repr(self):
        fmt_str = ['OWN:']
        if self.norm_groups > 1:
            fmt_str.append('groups={}'.format(self.norm_groups))
        return ', '.join(fmt_str)


class OWN_Conv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 norm_groups=1, NScale=1.414, adjustScale=False):
        super(OWN_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        print('OWN_conv:----norm_groups:', norm_groups, '---NScale:', NScale, '---adjust:', adjustScale)
        self.weight_normalization = OWNNorm(norm_groups=norm_groups)

        self.scale_ = torch.ones(out_channels, 1, 1, 1).fill_(NScale)
        if adjustScale:
            self.WNScale = Parameter(self.scale_)
        else:
            # self.scale = Variable(self.scale, requires_grad=False)
            self.register_buffer('WNScale', self.scale_)


    def forward(self, input_f: torch.Tensor) -> torch.Tensor:
        weight_q = self.weight_normalization(self.weight)
        weight_q = weight_q * self.WNScale
        out = F.conv2d(input_f, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out


if __name__ == '__main__':
    oni_ = OWNNorm(norm_groups=2)
    print(oni_)
    w_ = torch.randn(4, 4, 3, 3)
    w_.requires_grad_()
    y_ = oni_(w_)
    z_ = y_.view(w_.size(0), -1)
    print(z_.matmul(z_.t()))

    y_.sum().backward()
    print('w grad', w_.grad.size())

