from abc import abstractmethod
import math

import numpy as np

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from modules.layers import Flatten, SigmoidFA

from helpers import similarity

import copy

from modules.optimisers import NetworkCostOptimiser


class ANN(nn.Module):
    def __init__(self, n_inputs, n_hidden_layers, n_hidden_units, n_outputs, device):
        super(ANN, self).__init__()

        self.linear_layers = []

        if n_hidden_layers == 0:
            self.linear_layers.append(nn.Linear(n_inputs, n_outputs))
            # self.classification_layers.append(nn.Sigmoid())
        else:
            # self.layers.append(ContinuousBurstCCNHiddenLayer(n_inputs, n_hidden_units, n_hidden_units, p_baseline, device))
            self.linear_layers.append(nn.Linear(n_inputs, n_hidden_units))
            # self.classification_layers.append(nn.Sigmoid())

            for i in range(1, n_hidden_layers):
                self.linear_layers.append(nn.Linear(n_hidden_units, n_hidden_units))
                # self.classification_layers.append(nn.Sigmoid())

            self.linear_layers.append(nn.Linear(n_hidden_units, n_outputs))
            # self.classification_layers.append(nn.Sigmoid())


        all_layers = []
        for l in self.linear_layers:
            all_layers.append(l)
            all_layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*all_layers)
        self.layers.to(device)

    def forward(self, x):
        x = x.view(-1)

        for layer in self.linear_layers:
            x = layer(x)
            # print(x)
            x = torch.sigmoid(x)

        return x
        # return self.layers(x)

    def set_weights(self, weight_list, bias_list):
        for i, (weights, biases) in enumerate(zip(weight_list, bias_list)):
            self.linear_layers[i].weight.data = weights.detach().clone()
            self.linear_layers[i].bias.data = biases.detach().clone()


class MNISTNetFA(nn.Module):
    def __init__(self, n_hidden_layers, n_hidden_units, Y_mode, Y_scale, device):
        super(MNISTNetFA, self).__init__()

        assert Y_mode in ['tied', 'symmetric_init', 'random_init']
        self.Y_feedback_mode = Y_mode
        self.Y_feedback_scale = Y_scale
        self.device = device

        self.Y_learning = False
        self.feature_layers = []

        self.feature_layers.append(Flatten())

        self.classification_layers = []

        if n_hidden_layers == 0:
            self.classification_layers.append(SigmoidFA(784, 10, device))
        if n_hidden_layers == 1:
            self.classification_layers.append(SigmoidFA(784, n_hidden_units, device))
            self.classification_layers.append(SigmoidFA(n_hidden_units, 10, device))
        else:
            self.classification_layers.append(SigmoidFA(784, n_hidden_units, device))

            for i in range(1, n_hidden_layers):
                self.classification_layers.append(SigmoidFA(n_hidden_units, n_hidden_units, device))

            self.classification_layers.append(SigmoidFA(n_hidden_units, 10, device))

        self.out = nn.Sequential(*(self.feature_layers + self.classification_layers))

        self._initialize_weights()

    def forward(self, x):
        self.output = self.out(x)
        return self.output

    def backward(self, target):
        feedback_bp, feedback_a = -(target - self.output), -(target - self.output)

        feedback_bp, feedback_fa = self.classification_layers[-1].backward(feedback_bp, feedback_a)
        for i in range(len(self.classification_layers) - 2, -1, -1):
            feedback_bp, feedback_fa = self.classification_layers[i].backward(feedback_bp, feedback_fa)

    def update_weights(self, lrs, momentum=0, weight_decay=0, optimiser=None, global_cost=None, batch_size=1):
        weight_grads = [self.classification_layers[i].grad_weight_bp.detach() for i in range(len(self.classification_layers))]
        bias_grads = [self.classification_layers[i].grad_bias_bp.detach() for i in range(len(self.classification_layers))]

        if isinstance(optimiser, NetworkCostOptimiser):
            optimiser.update_buffers(global_cost=global_cost)
        else:
            optimiser.update_buffers(weight_grads=weight_grads, bias_grads=bias_grads)

        weight_updates, bias_updates = optimiser.compute_updates(lrs=lrs, weight_grads=weight_grads, bias_grads=bias_grads)

        for i in range(len(self.classification_layers)):
            self.classification_layers[i].update_weights(weight_updates[i], bias_updates[i])

        # for i in range(len(self.classification_layers)):
        #     self.classification_layers[i].update_weights(lr=lrs[i], momentum=momentum,
        #                                                  weight_decay=weight_decay,
        #                                                  batch_size=batch_size)
        for i in range(1, len(self.classification_layers)):
            if self.Y_feedback_mode == 'tied':
                self.classification_layers[i].weight_Y = self.Y_feedback_scale * copy.deepcopy(self.classification_layers[i].weight)

    def loss(self, output, target):
        return F.mse_loss(output, target)

    def weight_angles_W_Y(self):
        weight_angles = []

        for i in range(1, len(self.classification_layers)):
            weight_angles.append((180 / math.pi) * torch.acos(
                similarity(self.classification_layers[i].weight.flatten(),
                           self.classification_layers[i].weight_Y.flatten())).item())

        return weight_angles

    def grad_angles(self):
        grad_angles = []

        for i in range(len(self.classification_layers)):
            a = (180.0 / math.pi) * (torch.acos(similarity(self.classification_layers[i].grad_weight_fa.flatten(),
                                                           self.classification_layers[
                                                               i].grad_weight_bp.flatten()))).item()

            if np.isnan(a):
                a = 90.0
                assert (self.classification_layers[i].grad_weight_fa == 0.0).all() or torch.any(
                    torch.isnan(self.classification_layers[i].grad_weight))
                import warnings

                if (self.classification_layers[i].grad_weight_fa == 0.0).all():
                    warnings.warn(f'Updates in layer {i} are 0!')
                elif torch.any(torch.isnan(self.classification_layers[i].grad_weight_fa)):
                    warnings.warn(f'Gradients in layer {i} are NaN!')

            grad_angles.append(a)

        return grad_angles

    def global_grad_angle(self):
        grad_weights = []
        grad_weight_bps = []
        for i in range(len(self.classification_layers)):
            grad_weights.append(self.classification_layers[i].grad_weight_fa.flatten())
            grad_weight_bps.append(self.classification_layers[i].grad_weight_bp.flatten())

        grad_weights = torch.cat(grad_weights)
        grad_weight_bps = torch.cat(grad_weight_bps)

        a = (180.0 / math.pi) * (torch.acos(similarity(grad_weights,
                                                       grad_weight_bps))).item()

        return a

    def grad_magnitudes(self):
        grad_magnitudes = []
        for i in range(len(self.classification_layers)):
            m = torch.mean(torch.abs(self.classification_layers[i].grad_weight_fa)).item()
            grad_magnitudes.append(m)

        return grad_magnitudes

    def bp_grad_magnitudes(self):
        bp_grad_magnitudes = []
        for i in range(len(self.classification_layers)):
            m = torch.mean(torch.abs(self.classification_layers[i].grad_weight_bp)).item()
            bp_grad_magnitudes.append(m)

        return bp_grad_magnitudes



    def _initialize_weights(self):
        self._initialize_ff_weights()
        self._initialize_secondary_weights()

    def _initialize_ff_weights(self):
        module_list = list(self.modules())

        for module_index, m in enumerate(module_list):
            if isinstance(m, SigmoidFA):
                nn.init.xavier_normal_(m.weight, gain=3.6)
                nn.init.constant_(m.bias, 0)

    def _initialize_secondary_weights(self):
        layer_index = 0
        module_list = list(self.modules())

        for module_index, m in enumerate(module_list):
            if isinstance(m, SigmoidFA):
                if self.Y_feedback_mode == 'tied' or self.Y_feedback_mode == 'symmetric_init':
                    m.weight_Y = self.Y_feedback_scale * copy.deepcopy(module_list[module_index].weight.detach())
                elif self.Y_feedback_mode == 'random_init':
                    init.normal_(m.weight_Y, 0, self.Y_feedback_scale)

                layer_index += 1

    def _initialize_weights_from_list(self, ff_weights_list, ff_bias_list):
        layer_index = 0
        for m in self.modules():
            if isinstance(m, SigmoidFA):
                m.weight = copy.deepcopy(ff_weights_list[layer_index].detach())
                m.bias = copy.deepcopy(ff_bias_list[layer_index].detach())
                layer_index += 1

        self._initialize_secondary_weights()


class BioNetwork(nn.Module):

    @abstractmethod
    def get_layer_states(self):
        pass

    @abstractmethod
    def get_weight_angles(self):
        pass

    @abstractmethod
    def get_gradient_magnitudes(self):
        pass
