import torch
import torch.nn as nn
import torch.nn.functional as F


class EDNOutputLayer(nn.Module):
    def __init__(self, in_features, out_features, lambda_output, device):
        super(EDNOutputLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.lambda_output = lambda_output

        self.forward_noise = None

        self.weight = torch.Tensor(out_features, in_features).to(device)
        self.bias = torch.Tensor(out_features).to(device)

        self.delta_weight = torch.zeros(out_features, in_features).to(device)
        self.delta_bias = torch.zeros(out_features).to(device)

    def forward(self, input):
        if self.forward_noise is not None:
            self.input = input + self.forward_noise * torch.randn(input.shape, device=input.device)
        else:
            self.input = input
        self.pyr_basal = F.linear(self.input, self.weight, self.bias)
        self.pyr_soma = self.pyr_basal

        self.pyr_basal_rate = torch.sigmoid(self.pyr_basal)
        self.pyr_soma_rate = torch.sigmoid(self.pyr_soma)

        return self.pyr_soma_rate

    def backward(self, target_rate):

        if target_rate is not None:
            softened_rate = 0.2 + target_rate * 0.6
            target_potentials = torch.log(softened_rate / (1.0 - softened_rate))
        else:
            target_potentials = self.pyr_basal

        self.pyr_soma_t = (1 - self.lambda_output) * self.pyr_basal + self.lambda_output * target_potentials
        self.pyr_soma_rate_t = torch.sigmoid(self.pyr_soma_t)

        self.delta = -(self.pyr_soma_rate_t - self.pyr_basal_rate)

        if target_rate is not None:
            self.delta_bp = -(target_rate - self.pyr_soma_rate) * self.pyr_soma_rate * (1 - self.pyr_soma_rate)
            self.delta_fa = -(target_rate - self.pyr_soma_rate) * self.pyr_soma_rate * (1 - self.pyr_soma_rate)
        else:
            self.delta_bp = 0.0 * self.pyr_soma_rate * (1 - self.pyr_soma_rate)
            self.delta_fa = 0.0 * self.pyr_soma_rate * (1 - self.pyr_soma_rate)

        batch_size = self.delta.shape[0]

        self.grad_weight = self.delta.transpose(0, 1).mm(self.input) / batch_size
        self.grad_bias = torch.sum(self.delta, dim=0) / batch_size

        # delta_weight = -torch.mean(torch.bmm(self.delta.unsqueeze(2), self.input.unsqueeze(1)), dim=0)
        # delta_bias = -torch.mean(error_pyr, dim=0)

        self.grad_weight_bp = self.delta_bp.transpose(0, 1).mm(self.input) / batch_size
        self.grad_bias_bp = torch.sum(self.delta_bp, dim=0) / batch_size

        self.grad_weight_fa = self.delta_fa.transpose(0, 1).mm(self.input) / batch_size
        self.grad_bias_fa = torch.sum(self.delta_fa, dim=0) / batch_size

        return self.pyr_soma_t, self.pyr_soma_rate_t, self.delta_bp.mm(self.weight), self.delta_fa

    def update_weights(self, weight_update, bias_update, weight_decay):
        self.weight.data += weight_update - weight_decay * self.weight.data
        self.bias.data += bias_update - weight_decay * self.bias.data

    def update_secondary_weights(self, lr_Y, lr_pyr_intn, lr_intn_pyr):
        pass

    # def update_weights(self, lr_ff, lr_Y, lr_pyr_intn, lr_intn_pyr, momentum=0, weight_decay=0, batch_size=1):
    #     self.delta_weight = -lr_ff * self.grad_weight / batch_size + momentum * self.delta_weight
    #     self.delta_bias = -lr_ff * self.grad_bias / batch_size + momentum * self.delta_bias
    #
    #     self.weight += self.delta_weight - weight_decay * self.weight
    #     self.bias += self.delta_bias

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class EDNHiddenLayer(nn.Module):
    def __init__(self, in_features, out_features, next_features, lambda_intn, lambda_hidden, Y_learning, device):
        super(EDNHiddenLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.lambda_intn = lambda_intn
        self.lambda_hidden = lambda_hidden

        self.Y_learning = Y_learning

        self.forward_noise = None

        self.weight = torch.Tensor(out_features, in_features).to(device)
        self.bias = torch.Tensor(out_features).to(device)

        self.weight_pyr_intn = torch.Tensor(next_features, out_features).to(device)
        self.bias_pyr_intn = torch.Tensor(next_features).to(device)

        self.weight_intn_pyr = torch.Tensor(next_features, out_features).to(device)

        self.weight_Y = torch.Tensor(next_features, out_features).to(device)

        self.delta_weight = torch.zeros(out_features, in_features).to(device)
        self.delta_bias = torch.zeros(out_features).to(device)

        self.delta_weight_pyr_intn = torch.zeros(next_features, out_features).to(device)
        self.delta_bias_pyr_intn = torch.zeros(next_features).to(device)

        self.delta_weight_intn_pyr = torch.zeros(next_features, out_features).to(device)
        self.delta_weight_Y = torch.zeros(next_features, out_features).to(device)


    def forward(self, input):
        if self.forward_noise is not None:
            self.input = input + self.forward_noise * torch.randn(input.shape, device=input.device)
        else:
            self.input = input

        self.pyr_basal = F.linear(self.input, self.weight, self.bias)
        self.pyr_soma = self.pyr_basal

        self.pyr_basal_rate = torch.sigmoid(self.pyr_basal)
        self.pyr_soma_rate = torch.sigmoid(self.pyr_soma)

        self.intn_basal = F.linear(self.pyr_soma_rate, self.weight_pyr_intn, self.bias_pyr_intn)
        self.intn_soma = self.intn_basal

        self.intn_basal_rate = torch.sigmoid(self.intn_basal)
        self.intn_soma_rate = torch.sigmoid(self.intn_soma)

        return self.pyr_soma_rate

    def backward(self, next_layer_pyr_soma_t, next_layer_pyr_soma_rate_t, bp_grad, fa_delta):

        self.intn_soma = (1 - self.lambda_intn) * self.intn_basal + self.lambda_intn * next_layer_pyr_soma_t
        self.intn_soma_rate = torch.sigmoid(self.intn_soma)

        self.pyr_apical = next_layer_pyr_soma_rate_t.mm(self.weight_Y) - self.intn_soma_rate.mm(self.weight_intn_pyr)

        self.pyr_soma_t = self.pyr_basal + self.lambda_hidden * self.pyr_apical
        self.pyr_soma_rate_t = torch.sigmoid(self.pyr_soma_t)

        self.delta = -(self.pyr_soma_rate_t - self.pyr_basal_rate)
        self.delta_bp = bp_grad * self.pyr_soma_rate * (1 - self.pyr_soma_rate)
        self.delta_fa = fa_delta.mm(self.weight_Y) * self.pyr_soma_rate * (1 - self.pyr_soma_rate)

        batch_size = self.delta.shape[0]

        self.grad_weight = self.delta.transpose(0, 1).mm(self.input) / batch_size
        self.grad_bias = torch.sum(self.delta, dim=0) / batch_size

        self.grad_weight_bp = self.delta_bp.transpose(0, 1).mm(self.input) / batch_size
        self.grad_bias_bp = torch.sum(self.delta_bp, dim=0) / batch_size

        self.grad_weight_fa = self.delta_fa.transpose(0, 1).mm(self.input) / batch_size
        self.grad_bias_fa = torch.sum(self.delta_fa, dim=0) / batch_size

        self.grad_weight_pyr_intn = -(self.intn_soma_rate - self.intn_basal_rate).transpose(0, 1).mm(self.pyr_soma_rate_t) / batch_size
        self.grad_bias_pyr_intn = -torch.sum(self.intn_soma_rate - self.intn_basal_rate, dim=0) / batch_size

        self.grad_weight_intn_pyr = -self.intn_soma_rate.transpose(0, 1).mm(self.pyr_apical) / batch_size

        return self.pyr_soma_t, self.pyr_soma_rate_t, self.delta_bp.mm(self.weight), self.delta_fa


    # def update_weights(self, lr_ff, lr_Y, lr_pyr_intn, lr_intn_pyr, momentum=0, weight_decay=0, batch_size=1):
    #     self.delta_ff_weight = -lr_ff * self.grad_ff_weight / batch_size + momentum * self.delta_ff_weight
    #     self.delta_ff_bias = -lr_ff * self.grad_ff_bias / batch_size + momentum * self.delta_ff_bias
    #
    #     self.delta_pyr_intn_weight = -lr_pyr_intn * self.grad_pyr_intn_weight / batch_size + momentum * self.delta_pyr_intn_weight
    #     self.delta_pyr_intn_bias = -lr_pyr_intn * self.grad_pyr_intn_bias / batch_size + momentum * self.delta_pyr_intn_bias
    #
    #     self.delta_intn_pyr_weight = -lr_intn_pyr * self.grad_intn_pyr_weight / batch_size + momentum * self.delta_intn_pyr_weight
    #
    #     self.ff_weight += self.delta_ff_weight - weight_decay * self.ff_weight
    #     self.ff_bias += self.delta_ff_bias
    #
    #     self.pyr_intn_weight += self.delta_pyr_intn_weight - weight_decay * self.pyr_intn_weight
    #     self.pyr_intn_bias += self.delta_pyr_intn_bias
    #
    #     self.intn_pyr_weight += self.delta_intn_pyr_weight - weight_decay * self.intn_pyr_weight

    def update_weights(self, weight_update, bias_update, weight_decay):
        self.weight.data += weight_update - weight_decay * self.weight.data
        self.bias.data += bias_update - weight_decay * self.bias.data

    def update_secondary_weights(self, lr_Y, lr_pyr_intn, lr_intn_pyr):
        self.delta_weight_pyr_intn = -lr_pyr_intn * self.grad_weight_pyr_intn
        self.delta_bias_pyr_intn = -lr_pyr_intn * self.grad_bias_pyr_intn

        self.delta_weight_intn_pyr = -lr_intn_pyr * self.grad_weight_intn_pyr

        self.weight_pyr_intn += self.delta_weight_pyr_intn
        self.bias_pyr_intn += self.delta_bias_pyr_intn

        self.weight_intn_pyr += self.delta_weight_intn_pyr

        if self.Y_learning:
            self.delta_weight_Y = -lr_Y * self.grad_weight_Y
            self.weight_Y += self.delta_weight_Y


    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )