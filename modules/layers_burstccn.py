import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable
import torch.distributions as tdist
from torch.utils.cpp_extension import load
# cudnn_convolution = load(name="cudnn_convolution", sources=["C:/Users/hz15605/Documents/burst-models/cudnn_convolution.cpp"], verbose=True)
use_cudnn = False


class BurstCCNOutputLayer(nn.Module):
    def __init__(self, in_features, out_features, p_baseline, weight_Y_learning, weight_Q_learning, device):
        super(BurstCCNOutputLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.p_baseline = p_baseline

        self.forward_noise = None
        # self.weight_Y_learning = weight_Y_learning
        # self.weight_Q_learning = weight_Q_learning

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features).to(device), requires_grad=False)
        self.bias = nn.Parameter(torch.Tensor(out_features).to(device), requires_grad=False)

        # self.weight_Y = torch.Tensor(out_features, in_features).to(device)
        # self.weight_Q = torch.Tensor(out_features, in_features).to(device)

        self.delta_weight = nn.Parameter(torch.zeros(out_features, in_features).to(device), requires_grad=False)
        self.delta_bias = nn.Parameter(torch.zeros(out_features).to(device), requires_grad=False)

        self.p = self.p_baseline * torch.ones(self.out_features).to(device)

        # if self.weight_Y_learning:
        #     self.delta_weight_Y = torch.zeros(out_features, in_features).to(device)
        #
        # if self.weight_Q_learning:
        #     self.delta_weight_Q = torch.zeros(out_features, in_features).to(device)

    def forward(self, input):
        if self.forward_noise is not None:
            self.input = input + self.forward_noise * torch.randn(input.shape, device=input.device)
        else:
            self.input = input
        self.e = torch.sigmoid(F.linear(self.input, self.weight, self.bias))
        # self.e = F.linear(self.input, self.weight, self.bias)
        return self.e

    def backward(self, b_input):
        # #TODO: remove
        if b_input is None:
            b_input = self.e

        self.p = self.p_baseline
        self.p_t = self.p_baseline * ((b_input - self.e) * (1 - self.e) + 1)

        self.b = self.p * self.e
        self.b_t = self.p_t * self.e

        self.delta = -(self.b_t - self.b)

        self.delta_bp = -(b_input - self.e) * self.e * (1 - self.e)
        self.delta_fa = -(b_input - self.e) * self.e * (1 - self.e)

        batch_size = self.delta.shape[0]

        self.grad_weight = self.delta.transpose(0, 1).mm(self.input) / batch_size
        self.grad_bias = torch.sum(self.delta, dim=0) / batch_size

        self.grad_weight_bp = self.delta_bp.transpose(0, 1).mm(self.input) / batch_size
        self.grad_bias_bp = torch.sum(self.delta_bp, dim=0) / batch_size

        self.grad_weight_fa = self.delta_fa.transpose(0, 1).mm(self.input) / batch_size
        self.grad_bias_fa = torch.sum(self.delta_fa, dim=0) / batch_size

        # if self.weight_Y_learning:
        #     delta = -(self.p_t - self.p_baseline) * self.e
        #     self.grad_weight_Y = delta.transpose(0, 1).mm(self.input)
        #
        # if self.weight_Q_learning:
        #     delta = -(self.p_t - self.p_baseline) * self.e
        #     self.grad_weight_Q = delta.transpose(0, 1).mm(self.input)

        return self.b_t, self.e, self.delta_bp.mm(self.weight), self.delta_fa

    def update_weights(self, weight_update, bias_update, weight_decay):
        self.weight.data += weight_update - weight_decay * self.weight.data
        self.bias.data += bias_update - weight_decay * self.bias.data

    def update_secondary_weights(self, lr_Q, lr_Y):
        pass

    # def update_weights(self, lr, lr_Y, lr_Q, momentum=0, weight_decay=0):
    #     self.delta_weight = -lr * self.grad_weight + momentum * self.delta_weight
    #     self.delta_bias = -lr * self.grad_bias + momentum * self.delta_bias
    #
    #     self.weight += self.delta_weight - weight_decay * self.weight
    #     self.bias += self.delta_bias

    # if self.weight_Y_learning:
    #     self.delta_weight_Y = -lr * self.grad_weight_Y / batch_size + momentum * self.delta_weight_Y
    #
    #     self.weight_Y += self.delta_weight_Y - weight_decay * self.weight_Y
    #
    # if self.weight_Q_learning:
    #     self.delta_weight_Q = -lr * self.grad_weight_Q / batch_size + momentum * self.delta_weight_Q
    #
    #     self.weight_Q += self.delta_weight_Q - weight_decay * self.weight_Q

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class BurstCCNHiddenLayer(nn.Module):
    def __init__(self, in_features, out_features, next_features, p_baseline, weight_Y_learning, weight_Q_learning,
                 device):
        super(BurstCCNHiddenLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.p_baseline = p_baseline

        self.forward_noise = None

        self.weight_Y_learning = weight_Y_learning
        self.weight_Q_learning = weight_Q_learning

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features).to(device), requires_grad=False)
        self.bias = nn.Parameter(torch.Tensor(out_features).to(device), requires_grad=False)

        self.weight_Y = nn.Parameter(torch.Tensor(next_features, out_features).to(device), requires_grad=False)
        self.weight_Q = nn.Parameter(torch.Tensor(next_features, out_features).to(device), requires_grad=False)

        self.delta_weight = nn.Parameter(torch.zeros(out_features, in_features).to(device), requires_grad=False)
        self.delta_bias = nn.Parameter(torch.zeros(out_features).to(device), requires_grad=False)

        if self.weight_Y_learning:
            self.delta_weight_Y = nn.Parameter(torch.zeros(next_features, out_features).to(device), requires_grad=False)

        if self.weight_Q_learning:
            self.delta_weight_Q = nn.Parameter(torch.zeros(next_features, out_features).to(device), requires_grad=False)

    def forward(self, input):
        if self.forward_noise is not None:
            self.input = input + self.forward_noise * torch.randn(input.shape, device=input.device)
        else:
            self.input = input
        self.e = torch.sigmoid(F.linear(self.input, self.weight, self.bias))
        return self.e

    def backward(self, b_t_input, e_input, b_input_bp, next_delta_fa):

        self.apic = b_t_input.mm(self.weight_Y) - e_input.mm(self.weight_Q)

        # self.p_t = torch.sigmoid(4.0 * self.apic * (1 - self.e) / torch.linalg.norm(self.apic))
        self.p_t = torch.sigmoid(4.0 * self.apic * (1 - self.e))

        self.b = self.p_baseline * self.e
        self.b_t = self.p_t * self.e

        self.delta = -(self.b_t - self.b)

        self.delta_bp = b_input_bp * self.e * (1 - self.e)
        self.delta_fa = next_delta_fa.mm(self.weight_Y) * self.e * (1 - self.e)

        batch_size = self.delta.shape[0]

        self.grad_weight = self.delta.transpose(0, 1).mm(self.input) / batch_size
        self.grad_bias = torch.sum(self.delta, dim=0) / batch_size

        self.grad_weight_bp = self.delta_bp.transpose(0, 1).mm(self.input) / batch_size
        self.grad_bias_bp = torch.sum(self.delta_bp, dim=0) / batch_size

        self.grad_weight_fa = self.delta_fa.transpose(0, 1).mm(self.input) / batch_size
        self.grad_bias_fa = torch.sum(self.delta_fa, dim=0) / batch_size

        if self.weight_Y_learning:
            delta = self.apic
            self.grad_weight_Y = e_input.transpose(0, 1).mm(delta) / batch_size

        if self.weight_Q_learning:
            delta = self.apic
            self.grad_weight_Q = -e_input.transpose(0, 1).mm(delta) / batch_size

        return self.b_t, self.e, self.delta_bp.mm(self.weight), self.delta_fa

    def update_weights(self, weight_update, bias_update, weight_decay):
        self.weight.data += weight_update - weight_decay * self.weight.data
        self.bias.data += bias_update - weight_decay * self.bias.data

    def update_secondary_weights(self, lr_Q, lr_Y):
        if self.weight_Y_learning:
            self.delta_weight_Y.data = -lr_Y * self.grad_weight_Y
            self.weight_Y.data += self.delta_weight_Y

        if self.weight_Q_learning:
            self.delta_weight_Q.data = -lr_Q * self.grad_weight_Q
            self.weight_Q.data += self.delta_weight_Q

    # def update_weights(self, lr, lr_Y, lr_Q, momentum=0, weight_decay=0):
    #     self.delta_weight = -lr * self.grad_weight + momentum * self.delta_weight
    #     self.delta_bias = -lr * self.grad_bias + momentum * self.delta_bias
    #
    #     self.weight += self.delta_weight - weight_decay * self.weight
    #     self.bias += self.delta_bias
    #
    #     if self.weight_Y_learning:
    #         self.delta_weight_Y = -lr_Y * self.grad_weight_Y + momentum * self.delta_weight_Y
    #         self.weight_Y += self.delta_weight_Y - weight_decay * self.weight_Y
    #
    #     if self.weight_Q_learning:
    #         self.delta_weight_Q = -lr_Q * self.grad_weight_Q + momentum * self.delta_weight_Q
    #         self.weight_Q += self.delta_weight_Q - weight_decay * self.weight_Q

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Conv2dBurstCCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, in_size, kernel_size, stride, padding, dilation,
                 output_padding, groups, bias, padding_mode, p_baseline, weight_Y_learning,
                 weight_Q_learning, device):
        super(Conv2dBurstCCNLayer, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode

        self.p_baseline = p_baseline

        self.weight_Y_learning = weight_Y_learning
        self.weight_Q_learning = weight_Q_learning

        self.weight = torch.Tensor(
            out_channels, in_channels // groups, *kernel_size).to(device)

        self.out_size = int((in_size - kernel_size[0] + 2 * padding[0]) / stride[0] + 1)

        if bias:
            self.bias = torch.Tensor(out_channels).to(device)
        else:
            self.register_parameter('bias', None)

        self.delta_weight = torch.zeros(self.weight.shape).to(device)
        self.delta_bias = torch.zeros(self.bias.shape).to(device)


    def forward(self, input):
        self.input = input

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)

            self.input_size = F.pad(input, expanded_padding, mode='circular').size()

            self.e = torch.sigmoid(F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                                            self.weight, self.bias, self.stride,
                                            _pair(0), self.dilation, self.groups))

            return self.e

        if use_cudnn:
            self.e = torch.sigmoid(cudnn_convolution.convolution(self.input, self.weight, self.bias, self.stride,
                                                                 self.padding, self.dilation, self.groups, False,
                                                                 False))
        else:
            self.e = torch.sigmoid(F.conv2d(self.input, self.weight, self.bias, self.stride,
                                            self.padding, self.dilation, self.groups))

        return self.e

    def update_weights(self, weight_update, bias_update, weight_decay):
        self.weight.data += weight_update - weight_decay * self.weight.data
        self.bias.data += bias_update - weight_decay * self.bias.data

    def update_secondary_weights(self, lr_Q, lr_Y):
        if self.weight_Y_learning:
            self.delta_weight_Y.data = -lr_Y * self.grad_weight_Y
            self.weight_Y.data += self.delta_weight_Y

        if self.weight_Q_learning:
            self.delta_weight_Q.data = -lr_Q * self.grad_weight_Q
            self.weight_Q.data += self.delta_weight_Q

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class Conv2dBurstCCNHiddenLayer(Conv2dBurstCCNLayer):
    def __init__(self, in_channels, out_channels, next_channels, in_size, kernel_size, next_kernel_size, stride, next_stride, padding, dilation,
                 output_padding, groups, bias, padding_mode, p_baseline, weight_Y_learning,
                 weight_Q_learning, device):
        super(Conv2dBurstCCNHiddenLayer, self).__init__(in_channels, out_channels, in_size, kernel_size, stride, padding, dilation,
                 output_padding, groups, bias, padding_mode, p_baseline, weight_Y_learning,
                 weight_Q_learning, device)

        next_kernel_size = _pair(next_kernel_size)
        self.next_stride = _pair(next_stride)
        self.weight_Y = torch.Tensor(
            next_channels, out_channels // groups, *next_kernel_size).to(device)

        self.weight_Q = torch.Tensor(
            next_channels, out_channels // groups, *next_kernel_size).to(device)

        if self.weight_Y_learning:
            self.delta_weight_Y = nn.Parameter(torch.zeros(self.weight_Y.shape).to(device), requires_grad=False)

        if self.weight_Q_learning:
            self.delta_weight_Q = nn.Parameter(torch.zeros(self.weight_Y.shape).to(device), requires_grad=False)

    def backward(self, b_t_input, e_input, b_input_bp, next_delta_fa):

        if use_cudnn:
            Y_input = cudnn_convolution.convolution_backward_input(self.e.shape, self.weight_Y, b_t_input, self.next_stride,
                                                                self.padding, self.dilation, self.groups, False,
                                                                False)
            Q_input = cudnn_convolution.convolution_backward_input(self.e.shape, self.weight_Q, e_input, self.next_stride,
                                                                self.padding, self.dilation, self.groups, False,
                                                                False)
        else:
            Y_input = nn.grad.conv2d_input(self.e.shape, self.weight_Y, b_t_input, self.next_stride,
                                                                   self.padding, self.dilation, self.groups)
            Q_input = nn.grad.conv2d_input(self.e.shape, self.weight_Q, e_input, self.next_stride,
                                                                   self.padding, self.dilation, self.groups)

        self.apic = Y_input - Q_input

        # self.p_t = torch.sigmoid(4.0 * self.apic * (1 - self.e) / torch.linalg.norm(self.apic))
        self.p_t = torch.sigmoid(4.0 * self.apic * (1 - self.e))

        self.b = self.p_baseline * self.e
        self.b_t = self.p_t * self.e

        self.delta = -(self.b_t - self.b)

        self.delta_bp = b_input_bp * self.e * (1 - self.e)

        if use_cudnn:
            self.delta_fa = cudnn_convolution.convolution_backward_input(self.e.shape, self.weight_Y, next_delta_fa, self.next_stride,
                                                                self.padding, self.dilation, self.groups, False,
                                                                False) * self.e * (1 - self.e)
        else:
            self.delta_fa = nn.grad.conv2d_input(self.e.shape, self.weight_Y, next_delta_fa, self.next_stride,
                                                                   self.padding, self.dilation, self.groups) * self.e * (1 - self.e)


        batch_size = self.delta.shape[0]

        if use_cudnn:
            self.grad_weight = cudnn_convolution.convolution_backward_weight(self.input, self.weight.shape, self.delta,
                                                                             self.stride, self.padding, self.dilation,
                                                                             self.groups, False, False) / batch_size
            self.grad_weight_bp = cudnn_convolution.convolution_backward_weight(self.input, self.weight.shape, self.delta_bp,
                                                                             self.stride, self.padding, self.dilation,
                                                                             self.groups, False, False) / batch_size
            self.grad_weight_fa = cudnn_convolution.convolution_backward_weight(self.input, self.weight.shape,
                                                                                self.delta_fa,
                                                                                self.stride, self.padding,
                                                                                self.dilation,
                                                                                self.groups, False, False) / batch_size
        else:
            self.grad_weight = nn.grad.conv2d_weight(self.input, self.weight.shape, self.delta, self.stride,
                                                     self.padding, self.dilation, self.groups) / batch_size
            self.grad_weight_bp = nn.grad.conv2d_weight(self.input, self.weight.shape, self.delta_bp, self.stride,
                                                     self.padding, self.dilation, self.groups) / batch_size
            self.grad_weight_fa = nn.grad.conv2d_weight(self.input, self.weight.shape, self.delta_fa, self.stride,
                                                     self.padding, self.dilation, self.groups) / batch_size

        self.grad_bias = torch.sum(self.delta, dim=[0, 2, 3]) / batch_size
        self.grad_bias_bp = torch.sum(self.delta_bp, dim=[0, 2, 3]) / batch_size
        self.grad_bias_fa = torch.sum(self.delta_fa, dim=[0, 2, 3]) / batch_size


        if self.weight_Y_learning:
            delta = self.apic
            self.grad_weight_Y = e_input.transpose(0, 1).mm(delta) / batch_size

        if self.weight_Q_learning:
            delta = self.apic
            self.grad_weight_Q = -e_input.transpose(0, 1).mm(delta) / batch_size


        if use_cudnn:
            feedback_bp = cudnn_convolution.convolution_backward_input(self.input.shape, self.weight,
                                                                            self.delta_bp, self.stride, self.padding,
                                                                            self.dilation, self.groups, False, False)
        else:
            feedback_bp = nn.grad.conv2d_input(
                self.input.shape, self.weight, self.delta_bp, self.stride, self.padding, self.dilation, self.groups)


        return self.b_t, self.e, feedback_bp, self.delta_fa



class Conv2dBurstCCNFinalLayer(Conv2dBurstCCNLayer):
    def __init__(self, in_channels, out_channels, next_features, in_size, kernel_size, stride, padding, dilation,
                 output_padding, groups, bias, padding_mode, p_baseline, weight_Y_learning,
                 weight_Q_learning, device):
        super(Conv2dBurstCCNFinalLayer, self).__init__(in_channels, out_channels, in_size, kernel_size, stride, padding, dilation,
                 output_padding, groups, bias, padding_mode, p_baseline, weight_Y_learning,
                 weight_Q_learning, device)

        self.out_features = out_channels * self.out_size**2

        self.weight_Y = nn.Parameter(torch.Tensor(next_features, self.out_features).to(device), requires_grad=False)
        self.weight_Q = nn.Parameter(torch.Tensor(next_features, self.out_features).to(device), requires_grad=False)

        if self.weight_Y_learning:
            self.delta_weight_Y = nn.Parameter(torch.zeros(self.weight_Y.shape).to(device), requires_grad=False)

        if self.weight_Q_learning:
            self.delta_weight_Q = nn.Parameter(torch.zeros(self.weight_Y.shape).to(device), requires_grad=False)

    def forward(self, input):
        output = super(Conv2dBurstCCNFinalLayer, self).forward(input)
        output_size = output.size()
        return output.view(output_size[0], -1)

    def backward(self, b_t_input, e_input, b_input_bp, next_delta_fa):

        b_input_bp = b_input_bp.view(self.e.shape)

        self.apic = (b_t_input.mm(self.weight_Y) - e_input.mm(self.weight_Q)).view(self.e.shape)

        # self.p_t = torch.sigmoid(4.0 * self.apic * (1 - self.e) / torch.linalg.norm(self.apic))
        self.p_t = torch.sigmoid(4.0 * self.apic * (1 - self.e))

        self.b = self.p_baseline * self.e
        self.b_t = self.p_t * self.e

        self.delta = -(self.b_t - self.b)

        self.delta_bp = b_input_bp * self.e * (1 - self.e)
        self.delta_fa = (next_delta_fa.mm(self.weight_Y)).view(self.e.shape) * self.e * (1 - self.e)

        batch_size = self.delta.shape[0]

        if use_cudnn:
            self.grad_weight = cudnn_convolution.convolution_backward_weight(self.input, self.weight.shape, self.delta,
                                                                             self.stride, self.padding, self.dilation,
                                                                             self.groups, False, False) / batch_size
            self.grad_weight_bp = cudnn_convolution.convolution_backward_weight(self.input, self.weight.shape, self.delta_bp,
                                                                             self.stride, self.padding, self.dilation,
                                                                             self.groups, False, False) / batch_size
            self.grad_weight_fa = cudnn_convolution.convolution_backward_weight(self.input, self.weight.shape,
                                                                                self.delta_fa,
                                                                                self.stride, self.padding,
                                                                                self.dilation,
                                                                                self.groups, False, False) / batch_size
        else:
            self.grad_weight = nn.grad.conv2d_weight(self.input, self.weight.shape, self.delta, self.stride,
                                                     self.padding, self.dilation, self.groups) / batch_size
            self.grad_weight_bp = nn.grad.conv2d_weight(self.input, self.weight.shape, self.delta_bp, self.stride,
                                                     self.padding, self.dilation, self.groups) / batch_size
            self.grad_weight_fa = nn.grad.conv2d_weight(self.input, self.weight.shape, self.delta_fa, self.stride,
                                                     self.padding, self.dilation, self.groups) / batch_size

        self.grad_bias = torch.sum(self.delta, dim=[0, 2, 3]) / batch_size
        self.grad_bias_bp = torch.sum(self.delta_bp, dim=[0, 2, 3]) / batch_size
        self.grad_bias_fa = torch.sum(self.delta_fa, dim=[0, 2, 3]) / batch_size

        # self.grad_weight = self.delta.transpose(0, 1).mm(self.input) / batch_size
        # self.grad_bias = torch.sum(self.delta, dim=0) / batch_size
        #
        # self.grad_weight_bp = self.delta_bp.transpose(0, 1).mm(self.input) / batch_size
        # self.grad_bias_bp = torch.sum(self.delta_bp, dim=0) / batch_size
        #
        # self.grad_weight_fa = self.delta_fa.transpose(0, 1).mm(self.input) / batch_size
        # self.grad_bias_fa = torch.sum(self.delta_fa, dim=0) / batch_size

        if self.weight_Y_learning:
            delta = self.apic
            self.grad_weight_Y = e_input.transpose(0, 1).mm(delta) / batch_size

        if self.weight_Q_learning:
            delta = self.apic
            self.grad_weight_Q = -e_input.transpose(0, 1).mm(delta) / batch_size

        if use_cudnn:
            feedback_bp = cudnn_convolution.convolution_backward_input(self.input.shape, self.weight,
                                                                            self.delta_bp, self.stride, self.padding,
                                                                            self.dilation, self.groups, False, False)
        else:
            feedback_bp = nn.grad.conv2d_input(
                self.input.shape, self.weight, self.delta_bp, self.stride, self.padding, self.dilation, self.groups)

        return self.b_t, self.e, feedback_bp, self.delta_fa


class DLBurstCCNHiddenLayer(nn.Module):
    def __init__(self, in_features, out_features, next_features, p_baseline, weight_Y_learning, weight_Q_learning,
                 device):
        super(DLBurstCCNHiddenLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.p_baseline = p_baseline

        self.forward_noise = None

        self.weight_Y_learning = weight_Y_learning
        self.weight_Q_learning = weight_Q_learning

        self.weight = torch.Tensor(out_features, in_features).to(device)
        self.bias = torch.Tensor(out_features).to(device)

        self.weight_Y_to_SST = torch.Tensor(next_features, next_features).to(device)
        self.weight_Y_from_SST = torch.Tensor(next_features, out_features).to(device)
        # self.weight_Y_direct = torch.Tensor(next_features, out_features).to(device)
        self.weight_Y_to_VIP = torch.Tensor(next_features, next_features).to(device)
        self.weight_Y_VIP_to_SST = torch.Tensor(next_features, next_features).to(device)
        self.weight_Y_from_SST2 = torch.Tensor(next_features, out_features).to(device)

        self.bias_Y_from_SST2 = torch.Tensor(1, out_features).to(device)

        self.weight_Q_to_NDNF = torch.Tensor(next_features, next_features).to(device)
        self.weight_Q_from_NDNF = torch.Tensor(next_features, out_features).to(device)
        self.weight_Q_direct = torch.Tensor(next_features, out_features).to(device)

        self.delta_weight = torch.zeros(out_features, in_features).to(device)
        self.delta_bias = torch.zeros(out_features).to(device)

        if self.weight_Y_learning:
            self.delta_weight_Y_from_SST = torch.zeros(next_features, out_features).to(device)
            self.delta_weight_Y_direct = torch.zeros(next_features, out_features).to(device)

        if self.weight_Q_learning:
            self.delta_weight_Q_from_NDNF = torch.zeros(next_features, out_features).to(device)
            self.delta_weight_Q_direct = torch.zeros(next_features, out_features).to(device)

    def forward(self, input):
        if self.forward_noise is not None:
            self.input = input + self.forward_noise * torch.randn(input.shape, device=input.device)
        else:
            self.input = input
        self.e = torch.sigmoid(F.linear(self.input, self.weight, self.bias))
        return self.e

    def backward(self, b_t_input, e_input, b_input_bp):

        self.sst = b_t_input.mm(self.weight_Y_to_SST)
        self.vip = b_t_input.mm(self.weight_Y_to_VIP)
        self.sst2 = 1.0 - self.vip.mm(self.weight_Y_VIP_to_SST)

        self.burst_input = torch.ones(1, b_t_input.shape[1], device=b_t_input.device).mm(
            self.weight_Y_from_SST2) - self.sst2.mm(self.weight_Y_from_SST2) - self.sst.mm(self.weight_Y_from_SST)
        # self.burst_input = b_t_input.mm(self.weight_Y_from_SST2) - self.sst.mm(self.weight_Y_from_SST)

        self.ndnf = e_input.mm(self.weight_Q_to_NDNF)
        self.event_input = e_input.mm(self.weight_Q_direct) - self.ndnf.mm(self.weight_Q_from_NDNF)

        self.apic = self.burst_input + self.event_input

        self.p_t = torch.sigmoid(4.0 * self.apic * (1 - self.e))

        self.b = self.p_baseline * self.e
        self.b_t = self.p_t * self.e

        self.delta = -(self.b_t - self.b)

        self.delta_bp = b_input_bp * self.e * (1 - self.e)

        self.grad_weight = self.delta.transpose(0, 1).mm(self.input)
        self.grad_bias = torch.sum(self.delta, dim=0)

        self.grad_weight_bp = self.delta_bp.transpose(0, 1).mm(self.input)
        self.grad_bias_bp = torch.sum(self.delta_bp, dim=0)

        if self.weight_Y_learning:
            delta = self.apic
            self.grad_weight_Y_direct = e_input.transpose(0, 1).mm(delta)
            self.grad_weight_Y_from_SST = -self.sst.transpose(0, 1).mm(delta)

        if self.weight_Q_learning:
            delta = self.apic
            self.grad_weight_Q_direct = e_input.transpose(0, 1).mm(delta)
            self.grad_weight_Q_from_NDNF = -self.ndnf.transpose(0, 1).mm(delta)

        return self.b_t, self.e, self.delta_bp.mm(self.weight)

    def update_weights(self, lr, lr_Y, lr_Q, momentum=0, weight_decay=0, batch_size=1):
        self.delta_weight = -lr * self.grad_weight / batch_size + momentum * self.delta_weight
        self.delta_bias = -lr * self.grad_bias / batch_size + momentum * self.delta_bias

        self.weight += self.delta_weight - weight_decay * self.weight
        self.bias += self.delta_bias

        if self.weight_Y_learning:
            self.delta_weight_Y_direct = -lr_Y * self.grad_weight_Y_direct / batch_size + momentum * self.delta_weight_Y_direct
            self.weight_Y_direct += self.delta_weight_Y_direct - weight_decay * self.weight_Y_direct

            self.delta_weight_Y_from_SST = -lr_Y * self.grad_weight_Y_from_SST / batch_size + momentum * self.delta_weight_Y_from_SST
            self.weight_Y_from_SST += self.delta_weight_Y_from_SST - weight_decay * self.weight_Y_from_SST

        if self.weight_Q_learning:
            self.delta_weight_Q_direct = -lr_Q * self.grad_weight_Q_direct / batch_size + momentum * self.delta_weight_Q_direct
            self.weight_Q_direct += self.delta_weight_Q_direct - weight_decay * self.weight_Q_direct

            self.weight_Q_direct[self.weight_Q_direct <= 0.0] = 0.0

            self.delta_weight_Q_from_NDNF = -lr_Q * self.grad_weight_Q_from_NDNF / batch_size + momentum * self.delta_weight_Q_from_NDNF
            self.weight_Q_from_NDNF += self.delta_weight_Q_from_NDNF - weight_decay * self.weight_Q_from_NDNF

            self.weight_Q_from_NDNF[self.weight_Q_from_NDNF <= 0.0] = 0.0

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
