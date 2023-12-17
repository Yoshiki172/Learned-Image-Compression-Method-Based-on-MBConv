import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Function


class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones(inputs.size()) * bound
        b = b.to(inputs.device)
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)
        
    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors
        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class GDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
    """

    def __init__(self,
                 ch,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=0.1,
                 reparam_offset=2**-18):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = reparam_offset

        self.build(ch)

    def build(self, ch):
        self.pedestal = self.reparam_offset**2
        self.beta_bound = ((self.beta_min + self.reparam_offset**2)**0.5)
        self.gamma_bound = self.reparam_offset

        # Create beta param
        beta = torch.sqrt(torch.ones(ch)+self.pedestal)
        self.beta = nn.Parameter(beta)

        # Create gamma param
        eye = torch.eye(ch)
        g = self.gamma_init*eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)

        self.gamma = nn.Parameter(gamma)
        self.pedestal = self.pedestal

    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size() 
            inputs = inputs.view(bs, ch, d*w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta**2 - self.pedestal

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma**2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs**2, gamma, beta)
        norm_ = torch.sqrt(norm_)

        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs

class GSDN(nn.Module):
    """Generalized Subtractive and Divisive Normalization layer.
    y[i] = (x[i] - )/ sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
    """
  
    def __init__(self,
                 ch,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=.1,
                 reparam_offset=2**-18):
        super().__init__()
        self.inverse = inverse
        self.build(ch, beta_min, gamma_init, reparam_offset)
  
    def build(self, ch, beta_min, gamma_init, reparam_offset):
        self.pedestal = reparam_offset**2
        self.beta_bound = torch.FloatTensor([ ( beta_min + reparam_offset**2)**.5 ] )
        self.gamma_bound = torch.FloatTensor( [ reparam_offset] )
        
        ###### param for divisive ######
        # Create beta param
        beta = torch.sqrt(torch.ones(ch)+self.pedestal)
        self.beta = nn.Parameter(beta)
        # Create gamma param
        eye = torch.eye(ch)
        g = gamma_init*eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)
        self.gamma = nn.Parameter(gamma)
        
        ###### param for subtractive ######
        # Create beta2 param
        beta2 = torch.zeros(ch)
        self.beta2 = nn.Parameter(beta2)
        # Create gamma2 param
        eye = torch.eye(ch)
        g = gamma_init*eye
        g = g + self.pedestal
        gamma2 = torch.sqrt(g)
        self.gamma2 = nn.Parameter(gamma2)

    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size() 
            inputs = inputs.view(bs, ch, d*w, h)

        _, ch, _, _ = inputs.size()
        
        if self.inverse:
            # Scale
            beta = LowerBound.apply(self.beta, self.beta_bound)
            beta = beta**2 - self.pedestal 
            gamma = LowerBound.apply(self.gamma, self.gamma_bound)
            gamma = gamma**2 - self.pedestal
            gamma = gamma.view(ch, ch, 1, 1)
            norm_ = nn.functional.conv2d(inputs**2, gamma, beta)
            norm_ = torch.sqrt(norm_)
      
            outputs = inputs * norm_  # modified
            
            # Mean
            beta2 = LowerBound.apply(self.beta2, self.beta_bound)
            beta2 = beta2**2 - self.pedestal 
            gamma2 = LowerBound.apply(self.gamma2, self.gamma_bound)
            gamma2 = gamma2**2 - self.pedestal
            gamma2 = gamma2.view(ch, ch, 1, 1)
            mean_ = nn.functional.conv2d(inputs, gamma2, beta2)
      
            outputs = outputs + mean_
        else:
            # Mean
            beta2 = LowerBound.apply(self.beta2, self.beta_bound)
            beta2 = beta2**2 - self.pedestal 
            gamma2 = LowerBound.apply(self.gamma2, self.gamma_bound)
            gamma2 = gamma2**2 - self.pedestal
            gamma2 = gamma2.view(ch, ch, 1, 1)
            mean_ = nn.functional.conv2d(inputs, gamma2, beta2)
      
            outputs = inputs - mean_  # modified

            # Scale
            beta = LowerBound.apply(self.beta, self.beta_bound)
            beta = beta**2 - self.pedestal 
            gamma = LowerBound.apply(self.gamma, self.gamma_bound)
            gamma = gamma**2 - self.pedestal
            gamma = gamma.view(ch, ch, 1, 1)
            norm_ = nn.functional.conv2d(inputs**2, gamma, beta)
            norm_ = torch.sqrt(norm_)
      
            outputs = outputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)

        return outputs