import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        self.input_size = x.size()
        return x.view(self.input_size[0], -1)

    def backward(self, b_input, b_input_t, b_input_bp, b_input_fa):
        return b_input.view(self.input_size), b_input_t.view(self.input_size), b_input_bp.view(self.input_size), b_input_fa.view(self.input_size)


class SigmoidFA(nn.Module):
    def __init__(self, in_features, out_features, device):
        super(SigmoidFA, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = torch.Tensor(out_features, in_features).to(device)
        self.bias = torch.Tensor(out_features).to(device)

        self.weight_Y = torch.Tensor(out_features, in_features).to(device)

        self.delta_weight = torch.zeros(out_features, in_features).to(device)
        self.delta_bias = torch.zeros(out_features).to(device)

    def forward(self, input):
        self.input = input
        self.output = F.sigmoid(F.linear(self.input, self.weight, self.bias))
        return self.output

    def backward(self, b_input_bp, b_input_fa):
        # self.delta_bp = -(b_input - self.output) * self.output * (1 - self.output)
        # self.delta_fa = -(b_input - self.output) * self.output * (1 - self.output)

        self.delta_bp = b_input_bp * self.output * (1 - self.output)
        self.delta_fa = b_input_fa * self.output * (1 - self.output)

        self.grad_weight_bp = self.delta_bp.transpose(0, 1).mm(self.input)
        self.grad_bias_bp = torch.sum(self.delta_bp, dim=0)

        self.grad_weight_fa = self.delta_fa.transpose(0, 1).mm(self.input)
        self.grad_bias_fa = torch.sum(self.delta_fa, dim=0)

        return self.delta_bp.mm(self.weight), self.delta_fa.mm(self.weight_Y)

    def update_weights(self, weight_update, bias_update):
        self.weight.data += weight_update
        self.bias.data += bias_update

    # def update_weights(self, lr, momentum=0.0, weight_decay=0.0, batch_size=1):
    #     self.delta_weight = -lr * self.grad_weight_fa / batch_size + momentum * self.delta_weight
    #     self.delta_bias = -lr * self.grad_bias_fa / batch_size + momentum * self.delta_bias
    #
    #     self.weight += self.delta_weight - weight_decay * self.weight
    #     self.bias += self.delta_bias

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )