import copy

import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from modules.layers import Flatten
from modules.layers_burstccn import BurstCCNHiddenLayer, BurstCCNOutputLayer, DLBurstCCNHiddenLayer, \
    Conv2dBurstCCNHiddenLayer, Conv2dBurstCCNFinalLayer
from modules.optimisers import AdamOptimiser, SGDOptimiser, SGDMomentumOptimiser, NeuronLeakOptimiser, \
    NetworkLeakOptimiser, NetworkCostOptimiser

import wandb

from helpers import similarity, generate_positive_full_rank_matrix


class BurstCCN(nn.Module):
    def __init__(self, n_inputs, n_outputs, p_baseline, n_hidden_layers, n_hidden_units, Y_mode, Q_mode, Y_scale,
                 Q_scale, Y_learning,
                 Q_learning, device):
        super(BurstCCN, self).__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.p_baseline = p_baseline

        assert Y_mode in ['tied', 'symmetric_init', 'random_init']
        self.Y_mode = Y_mode
        self.Y_scale = Y_scale

        assert Q_mode in ['tied', 'symmetric_init', 'random_init', 'W_symmetric_init']
        self.Q_mode = Q_mode
        self.Q_scale = Q_scale

        self.Y_learning = Y_learning
        self.Q_learning = Q_learning

        self.device = device

        self.feature_layers = []

        self.feature_layers.append(Flatten())

        self.classification_layers = []

        if n_hidden_layers == 0:
            self.classification_layers.append(
                BurstCCNOutputLayer(n_inputs, n_outputs, p_baseline, Y_learning, Q_learning, device))
        elif n_hidden_layers == 1:
            self.classification_layers.append(
                BurstCCNHiddenLayer(n_inputs, n_hidden_units, n_outputs, p_baseline, Y_learning, Q_learning, device))
            self.classification_layers.append(
                BurstCCNOutputLayer(n_hidden_units, n_outputs, p_baseline, Y_learning, Q_learning, device))
        else:
            self.classification_layers.append(
                BurstCCNHiddenLayer(n_inputs, n_hidden_units, n_hidden_units, p_baseline, Y_learning,
                                    Q_learning, device))

            for i in range(1, n_hidden_layers - 1):
                self.classification_layers.append(
                    BurstCCNHiddenLayer(n_hidden_units, n_hidden_units, n_hidden_units, p_baseline, Y_learning,
                                        Q_learning, device))

            self.classification_layers.append(
                BurstCCNHiddenLayer(n_hidden_units, n_hidden_units, n_outputs, p_baseline, Y_learning,
                                    Q_learning, device))
            self.classification_layers.append(
                BurstCCNOutputLayer(n_hidden_units, n_outputs, p_baseline, Y_learning, Q_learning, device))

        self.out = nn.Sequential(*(self.feature_layers + self.classification_layers))

        self._initialize_weights()

    # TODO: temp
    def set_forward_noise(self, forward_noise):
        for layer in self.classification_layers:
            layer.forward_noise = forward_noise

    def forward(self, x):
        return self.out(x)

    def backward(self, target):
        burst_rate, event_rate, feedback_bp, feedback_fa = self.classification_layers[-1].backward(target)
        for i in range(len(self.classification_layers) - 2, -1, -1):
            burst_rate, event_rate, feedback_bp, feedback_fa = self.classification_layers[i].backward(burst_rate,
                                                                                                      event_rate,
                                                                                                      feedback_bp,
                                                                                                      feedback_fa)

    def update_weights(self, lrs, lrs_Y, lrs_Q, optimiser, weight_decay=0, global_cost=None, use_backprop=False, use_feedback_alignment=False):
        assert not (use_backprop and use_feedback_alignment)

        if use_backprop:
            weight_grads = [self.classification_layers[i].grad_weight_bp.detach() for i in range(len(self.classification_layers))]
            bias_grads = [self.classification_layers[i].grad_bias_bp.detach() for i in range(len(self.classification_layers))]
        elif use_feedback_alignment:
            weight_grads = [self.classification_layers[i].grad_weight_fa.detach() for i in range(len(self.classification_layers))]
            bias_grads = [self.classification_layers[i].grad_bias_fa.detach() for i in range(len(self.classification_layers))]
        else:
            weight_grads = [self.classification_layers[i].grad_weight.detach() for i in range(len(self.classification_layers))]
            bias_grads = [self.classification_layers[i].grad_bias.detach() for i in range(len(self.classification_layers))]

        if isinstance(optimiser, NetworkCostOptimiser):
            optimiser.update_buffers(global_cost=global_cost)
        else:
            optimiser.update_buffers(weight_grads=weight_grads, bias_grads=bias_grads)

        weight_updates, bias_updates = optimiser.compute_updates(lrs=lrs,
                                                                 weight_grads=weight_grads,
                                                                 bias_grads=bias_grads)

        for i in range(len(self.classification_layers)):
            self.classification_layers[i].update_weights(weight_updates[i], bias_updates[i], weight_decay=weight_decay)
            self.classification_layers[i].update_secondary_weights(lr_Y=lrs_Y[i], lr_Q=lrs_Q[i])

        # for i in range(len(self.classification_layers)):
        #     self.classification_layers[i].update_weights(lr=lr[i], lr_Y=lr_Y[i], lr_Q=lr_Q[i], momentum=momentum,
        #                                                  weight_decay=weight_decay)

        for i in range(len(self.classification_layers)):
            if i < len(self.classification_layers) - 1:
                if self.Y_mode == 'tied':
                    self.classification_layers[i].weight_Y.data = self.Y_scale * copy.deepcopy(
                        self.classification_layers[i + 1].weight)
                if self.Q_mode == 'tied':
                    self.classification_layers[i].weight_Q.data = self.p_baseline * copy.deepcopy(
                        self.classification_layers[i].weight_Y)

    def loss(self, output, target):
        return F.mse_loss(output, target)

    def weight_angles_W_Y(self):
        weight_angles = []

        for i in range(1, len(self.classification_layers)):
            weight_angles.append((180 / math.pi) * torch.acos(
                similarity(self.classification_layers[i].weight.flatten(),
                           self.classification_layers[i - 1].weight_Y.flatten())).item())

        return weight_angles

    def weight_angles_Q_Y(self):
        weight_angles = []

        for i in range(len(self.classification_layers) - 1):
            weight_angles.append((180.0 / math.pi) * torch.acos(
                similarity(self.classification_layers[i].weight_Y.flatten(),
                           self.classification_layers[i].weight_Q.flatten())).item())

        return weight_angles

    def global_weight_angle_Q_Y(self):
        Y_weights = []
        Q_weights = []

        for i in range(len(self.classification_layers) - 1):
            Y_weights.append(self.classification_layers[i].weight_Y.flatten())
            Q_weights.append(self.classification_layers[i].weight_Q.flatten())

        Y_weights = torch.cat(Y_weights)
        Q_weights = torch.cat(Q_weights)

        a = (180.0 / math.pi) * (torch.acos(similarity(Y_weights,
                                                       Q_weights))).item()

        return a

    def bp_angles(self):
        bp_angles = []

        for i in range(len(self.classification_layers)):
            # a = np.mean([(180.0 / math.pi) * (torch.acos(
            #     similarity(self.classification_layers[i].delta[j].flatten(),
            #                self.classification_layers[i].delta_bp[j].flatten()))).cpu()
            #              for j in range(self.classification_layers[i].delta.shape[0])])

            a = (180.0 / math.pi) * (torch.acos(similarity(self.classification_layers[i].grad_weight.flatten(),
                                                           self.classification_layers[
                                                               i].grad_weight_bp.flatten()))).item()

            if np.isnan(a):
                a = 90.0
                assert (self.classification_layers[i].grad_weight == 0.0).all() or torch.any(
                    torch.isnan(self.classification_layers[i].grad_weight))
                import warnings

                if (self.classification_layers[i].grad_weight == 0.0).all():
                    warnings.warn(f'Updates in layer {i} are 0!')
                elif torch.any(torch.isnan(self.classification_layers[i].grad_weight)):
                    warnings.warn(f'Gradients in layer {i} are NaN!')

            bp_angles.append(a)

        return bp_angles

    def global_bp_angle(self):
        grad_weights = []
        grad_weight_bps = []
        for i in range(len(self.classification_layers)):
            grad_weights.append(self.classification_layers[i].grad_weight.flatten())
            grad_weight_bps.append(self.classification_layers[i].grad_weight_bp.flatten())

        grad_weights = torch.cat(grad_weights)
        grad_weight_bps = torch.cat(grad_weight_bps)

        a = (180.0 / math.pi) * (torch.acos(similarity(grad_weights,
                                                       grad_weight_bps))).item()

        return a

    def fa_angles(self):
        fa_angles = []

        for i in range(len(self.classification_layers)):
            a = (180.0 / math.pi) * (torch.acos(similarity(self.classification_layers[i].grad_weight.flatten(),
                                                           self.classification_layers[
                                                               i].grad_weight_fa.flatten()))).item()
            fa_angles.append(a)

        return fa_angles

    def global_fa_angle(self):
        grad_weights = []
        grad_weight_fas = []
        for i in range(len(self.classification_layers)):
            grad_weights.append(self.classification_layers[i].grad_weight.flatten())
            grad_weight_fas.append(self.classification_layers[i].grad_weight_fa.flatten())

        grad_weights = torch.cat(grad_weights)
        grad_weight_fas = torch.cat(grad_weight_fas)

        a = (180.0 / math.pi) * (torch.acos(similarity(grad_weights,
                                                       grad_weight_fas))).item()

        return a

    def fa_to_bp_angles(self):
        fa_to_bp_angles = []

        for i in range(len(self.classification_layers)):
            a = (180.0 / math.pi) * (torch.acos(similarity(self.classification_layers[i].grad_weight_bp.flatten(),
                                                           self.classification_layers[
                                                               i].grad_weight_fa.flatten()))).item()
            fa_to_bp_angles.append(a)

        return fa_to_bp_angles

    def global_fa_to_bp_angle(self):
        grad_weight_bps = []
        grad_weight_fas = []
        for i in range(len(self.classification_layers)):
            grad_weight_bps.append(self.classification_layers[i].grad_weight_bp.flatten())
            grad_weight_fas.append(self.classification_layers[i].grad_weight_fa.flatten())

        grad_weight_bps = torch.cat(grad_weight_bps)
        grad_weight_fas = torch.cat(grad_weight_fas)

        a = (180.0 / math.pi) * (torch.acos(similarity(grad_weight_bps,
                                                       grad_weight_fas))).item()

        return a

    def bp_grad_magnitudes(self):
        bp_grad_magnitudes = []
        for i in range(len(self.classification_layers)):
            m = torch.mean(torch.abs(self.classification_layers[i].grad_weight_bp)).item()
            bp_grad_magnitudes.append(m)

        return bp_grad_magnitudes

    def grad_magnitudes(self):
        grad_magnitudes = []
        for i in range(len(self.classification_layers)):
            m = torch.mean(torch.abs(self.classification_layers[i].grad_weight)).item()
            grad_magnitudes.append(m)

        return grad_magnitudes

    def log_layer_states(self):
        # Log hidden
        for i in range(len(self.classification_layers) - 1):
            layer = self.classification_layers[i]
            wandb.log({f"hidden{i}.event_rate": wandb.Histogram(layer.e.flatten().cpu().numpy()),
                       f"hidden{i}.apical": wandb.Histogram(layer.apic.flatten().cpu().numpy()),
                       f"hidden{i}.burst_prob": wandb.Histogram(layer.p_t.flatten().cpu().numpy()),
                       f"hidden{i}.burst_rate": wandb.Histogram(layer.b_t.flatten().cpu().numpy())}, commit=False)

        # Log output
        output_layer = self.classification_layers[-1]
        wandb.log({"output.event_rate": wandb.Histogram(output_layer.e.flatten().cpu().numpy()),
                   "output.burst_prob": wandb.Histogram(output_layer.p_t.flatten().cpu().numpy()),
                   "output.burst_rate": wandb.Histogram(output_layer.b_t.flatten().cpu().numpy())}, commit=False)

    def _initialize_weights(self):
        self._initialize_ff_weights()
        self._initialize_secondary_weights()

    def _initialize_ff_weights(self):
        module_list = list(self.modules())

        for module_index, m in enumerate(module_list):
            if isinstance(m, BurstCCNHiddenLayer) or isinstance(m, BurstCCNOutputLayer):
                nn.init.xavier_normal_(m.weight, gain=3.6)
                # nn.init.normal_(m.weight, 0.0, 0.1)
                nn.init.constant_(m.bias, 0)
                # m.weight.data.normal_(0, 0.1)
                # stdv = 1. / math.sqrt(m.weight.size(1))
                # m.bias.data.uniform_(-stdv, stdv)

    def _initialize_secondary_weights(self):
        layer_index = 0
        module_list = list(self.modules())

        for module_index, m in enumerate(module_list):
            if isinstance(m, BurstCCNHiddenLayer):
                if self.Y_mode == 'tied' or self.Y_mode == 'symmetric_init':
                    m.weight_Y.data = self.Y_scale * copy.deepcopy(module_list[module_index + 1].weight.detach())
                elif self.Y_mode == 'random_init':
                    init.normal_(m.weight_Y, 0, self.Y_scale)

                if self.Q_mode == 'tied' or self.Q_mode == 'symmetric_init':
                    assert self.p_baseline == 0.5
                    assert self.Q_scale == 1.0
                    m.weight_Q.data = self.p_baseline * copy.deepcopy(m.weight_Y.data)
                elif self.Q_mode == 'W_symmetric_init':
                    m.weight_Q.data = self.p_baseline * self.Q_scale * copy.deepcopy(module_list[module_index + 1].weight.detach())
                elif self.Q_mode == 'random_init':
                    init.normal_(m.weight_Q, 0, self.Q_scale)

                layer_index += 1

    def _initialize_weights_from_list(self, ff_weights_list, ff_bias_list):
        layer_index = 0
        for m in self.modules():
            if isinstance(m, BurstCCNHiddenLayer) or isinstance(m, BurstCCNOutputLayer):
                m.weight.data = copy.deepcopy(ff_weights_list[layer_index].detach())
                m.bias.data = copy.deepcopy(ff_bias_list[layer_index].detach())
                # if isinstance(m, BurstCCNHiddenLayer):
                #     # init.normal_(m.weight_Y, 0, self.weight_Y_std)
                #     m.weight_Y = copy.deepcopy(ff_weights_list[layer_index + 1].detach())
                #     m.weight_Q = 0.5 * copy.deepcopy(m.weight_Y)
                #
                # if isinstance(m, BurstCCNHiddenLayer) or isinstance(m, BurstCCNOutputLayer):
                layer_index += 1

        self._initialize_secondary_weights()


class ConvBurstCCN(nn.Module):
    def __init__(self, n_inputs, n_outputs, p_baseline, n_hidden_layers, n_hidden_units, Y_mode, Q_mode, Y_scale,
                 Q_scale, Y_learning,
                 Q_learning, device):
        super(ConvBurstCCN, self).__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.p_baseline = p_baseline

        assert Y_mode in ['tied', 'symmetric_init', 'random_init']
        self.Y_mode = Y_mode
        self.Y_scale = Y_scale

        assert Q_mode in ['tied', 'symmetric_init', 'random_init', 'W_symmetric_init']
        self.Q_mode = Q_mode
        self.Q_scale = Q_scale

        self.Y_learning = Y_learning
        self.Q_learning = Q_learning

        self.device = device

        self.classification_layers = []

        # self.classification_layers.append(
        #     Conv2dBurstCCNHiddenLayer(in_channels=1, out_channels=8, next_channels=16, in_size=28, kernel_size=4, next_kernel_size=3,
        #                               stride=2, padding=0, dilation=1,
        #                               output_padding=_pair(0), groups=1, bias=True, padding_mode='zeros',
        #                               p_baseline=self.p_baseline, weight_Y_learning=False,
        #                               weight_Q_learning=False, device=self.device))
        # # self.classification_layers.append(
        # #     Conv2dBurstCCNHiddenLayer(in_channels=64, out_channels=256, next_channels=256,
        # #                               in_size=self.classification_layers[0].out_size, kernel_size=5,
        # #                               stride=2, padding=0, dilation=1,
        # #                               output_padding=_pair(0), groups=1, bias=True, padding_mode='zeros',
        # #                               p_baseline=self.p_baseline, weight_Y_learning=False,
        # #                               weight_Q_learning=False, device=self.device))
        # self.classification_layers.append(
        #     Conv2dBurstCCNFinalLayer(in_channels=8, out_channels=16, next_features=500,
        #                              in_size=self.classification_layers[0].out_size, kernel_size=3,
        #                              stride=2, padding=0, dilation=1,
        #                              output_padding=_pair(0), groups=1, bias=True, padding_mode='zeros',
        #                              p_baseline=self.p_baseline, weight_Y_learning=False,
        #                              weight_Q_learning=False, device=self.device))
        #
        #
        # self.classification_layers.append(
        #     BurstCCNHiddenLayer(576, 500, 500, p_baseline, Y_learning,
        #                         Q_learning, device))
        # self.classification_layers.append(
        #     BurstCCNHiddenLayer(500, 500, 10, p_baseline, Y_learning,
        #                         Q_learning, device))
        # self.classification_layers.append(
        #     BurstCCNOutputLayer(500, 10, p_baseline, Y_learning, Q_learning, device))

        self.classification_layers.append(
            Conv2dBurstCCNHiddenLayer(in_channels=3, out_channels=64, next_channels=128, in_size=32, kernel_size=5, next_kernel_size=5,
                                      stride=2, next_stride=2, padding=0, dilation=1,
                                      output_padding=_pair(0), groups=1, bias=True, padding_mode='zeros',
                                      p_baseline=self.p_baseline, weight_Y_learning=False,
                                      weight_Q_learning=False, device=self.device))
        self.classification_layers.append(
            Conv2dBurstCCNHiddenLayer(in_channels=64, out_channels=128, next_channels=256, in_size=self.classification_layers[-1].out_size, kernel_size=5,  next_kernel_size=3,
                                      stride=2, next_stride=1, padding=0, dilation=1,
                                      output_padding=_pair(0), groups=1, bias=True, padding_mode='zeros',
                                      p_baseline=self.p_baseline, weight_Y_learning=False,
                                      weight_Q_learning=False, device=self.device))
        self.classification_layers.append(
            Conv2dBurstCCNFinalLayer(in_channels=128, out_channels=256, next_features=1480, in_size=self.classification_layers[-1].out_size, kernel_size=3,
                                     stride=1, padding=0, dilation=1,
                                     output_padding=_pair(0), groups=1, bias=True, padding_mode='zeros',
                                     p_baseline=self.p_baseline, weight_Y_learning=False,
                                     weight_Q_learning=False, device=self.device))


        self.classification_layers.append(
            BurstCCNHiddenLayer(2304, 1480, 10, p_baseline, Y_learning,
                                Q_learning, device))

        self.classification_layers.append(
            BurstCCNOutputLayer(1480, 10, p_baseline, Y_learning, Q_learning, device))

        self.out = nn.Sequential(*(self.classification_layers))

        self._initialize_weights()

    # TODO: temp
    def set_forward_noise(self, forward_noise):
        for layer in self.classification_layers:
            layer.forward_noise = forward_noise

    def forward(self, x):
        return self.out(x)

    def backward(self, target):
        burst_rate, event_rate, feedback_bp, feedback_fa = self.classification_layers[-1].backward(target)
        for i in range(len(self.classification_layers) - 2, -1, -1):
            burst_rate, event_rate, feedback_bp, feedback_fa = self.classification_layers[i].backward(burst_rate,
                                                                                                      event_rate,
                                                                                                      feedback_bp,
                                                                                                      feedback_fa)

        # for i in range(len(self.feature_layers) - 1, -1, -1):
        #     burst_rate, event_rate, feedback_bp, feedback_fa = self.feature_layers[i].backward(burst_rate,
        #                                                                                        event_rate,
        #                                                                                        feedback_bp,
        #                                                                                        feedback_fa)

    def update_weights(self, lrs, lrs_Y, lrs_Q, optimiser, momentum=0, weight_decay=0, global_cost=None, use_backprop=False, use_feedback_alignment=False):
        assert not (use_backprop and use_feedback_alignment)

        if use_backprop:
            weight_grads = [self.classification_layers[i].grad_weight_bp.detach() for i in range(len(self.classification_layers))]
            bias_grads = [self.classification_layers[i].grad_bias_bp.detach() for i in range(len(self.classification_layers))]
        elif use_feedback_alignment:
            weight_grads = [self.classification_layers[i].grad_weight_fa.detach() for i in range(len(self.classification_layers))]
            bias_grads = [self.classification_layers[i].grad_bias_fa.detach() for i in range(len(self.classification_layers))]
        else:
            weight_grads = [self.classification_layers[i].grad_weight.detach() for i in range(len(self.classification_layers))]
            bias_grads = [self.classification_layers[i].grad_bias.detach() for i in range(len(self.classification_layers))]

        if isinstance(optimiser, NetworkCostOptimiser):
            optimiser.update_buffers(global_cost=global_cost)
        else:
            optimiser.update_buffers(weight_grads=weight_grads, bias_grads=bias_grads)

        weight_updates, bias_updates = optimiser.compute_updates(lrs=lrs,
                                                                 weight_grads=weight_grads,
                                                                 bias_grads=bias_grads)

        for i in range(len(self.classification_layers)):
            self.classification_layers[i].update_weights(weight_updates[i], bias_updates[i], weight_decay)
            self.classification_layers[i].update_secondary_weights(lr_Y=lrs_Y[i], lr_Q=lrs_Q[i])


        # for i in range(len(self.classification_layers)):
        #     self.classification_layers[i].update_weights(lr=lr[i], lr_Y=lr_Y[i], lr_Q=lr_Q[i], momentum=momentum,
        #                                                  weight_decay=weight_decay)

        for i in range(len(self.classification_layers)):
            if i < len(self.classification_layers) - 1:
                if self.Y_mode == 'tied':
                    self.classification_layers[i].weight_Y.data = self.Y_scale * copy.deepcopy(
                        self.classification_layers[i + 1].weight)
                if self.Q_mode == 'tied':
                    self.classification_layers[i].weight_Q.data = self.p_baseline * copy.deepcopy(
                        self.classification_layers[i].weight_Y)

    def loss(self, output, target):
        return F.mse_loss(output, target)

    def weight_angles_W_Y(self):
        weight_angles = []

        for i in range(1, len(self.classification_layers)):
            weight_angles.append((180 / math.pi) * torch.acos(
                similarity(self.classification_layers[i].weight.flatten(),
                           self.classification_layers[i - 1].weight_Y.flatten())).item())

        return weight_angles

    def weight_angles_Q_Y(self):
        weight_angles = []

        for i in range(len(self.classification_layers) - 1):
            weight_angles.append((180.0 / math.pi) * torch.acos(
                similarity(self.classification_layers[i].weight_Y.flatten(),
                           self.classification_layers[i].weight_Q.flatten())).item())

        return weight_angles

    def global_weight_angle_Q_Y(self):
        Y_weights = []
        Q_weights = []

        for i in range(len(self.classification_layers) - 1):
            Y_weights.append(self.classification_layers[i].weight_Y.flatten())
            Q_weights.append(self.classification_layers[i].weight_Q.flatten())

        Y_weights = torch.cat(Y_weights)
        Q_weights = torch.cat(Q_weights)

        a = (180.0 / math.pi) * (torch.acos(similarity(Y_weights,
                                                       Q_weights))).item()

        return a

    def bp_angles(self):
        bp_angles = []

        for i in range(len(self.classification_layers)):
            # a = np.mean([(180.0 / math.pi) * (torch.acos(
            #     similarity(self.classification_layers[i].delta[j].flatten(),
            #                self.classification_layers[i].delta_bp[j].flatten()))).cpu()
            #              for j in range(self.classification_layers[i].delta.shape[0])])

            a = (180.0 / math.pi) * (torch.acos(similarity(self.classification_layers[i].grad_weight.flatten(),
                                                           self.classification_layers[
                                                               i].grad_weight_bp.flatten()))).item()

            if np.isnan(a):
                a = 90.0
                assert (self.classification_layers[i].grad_weight == 0.0).all() or torch.any(
                    torch.isnan(self.classification_layers[i].grad_weight))
                import warnings

                if (self.classification_layers[i].grad_weight == 0.0).all():
                    warnings.warn(f'Updates in layer {i} are 0!')
                elif torch.any(torch.isnan(self.classification_layers[i].grad_weight)):
                    warnings.warn(f'Gradients in layer {i} are NaN!')

            bp_angles.append(a)

        return bp_angles

    def global_bp_angle(self):
        grad_weights = []
        grad_weight_bps = []
        for i in range(len(self.classification_layers)):
            grad_weights.append(self.classification_layers[i].grad_weight.flatten())
            grad_weight_bps.append(self.classification_layers[i].grad_weight_bp.flatten())

        grad_weights = torch.cat(grad_weights)
        grad_weight_bps = torch.cat(grad_weight_bps)

        a = (180.0 / math.pi) * (torch.acos(similarity(grad_weights,
                                                       grad_weight_bps))).item()

        return a

    def fa_angles(self):
        fa_angles = []

        for i in range(len(self.classification_layers)):
            a = (180.0 / math.pi) * (torch.acos(similarity(self.classification_layers[i].grad_weight.flatten(),
                                                           self.classification_layers[
                                                               i].grad_weight_fa.flatten()))).item()
            fa_angles.append(a)

        return fa_angles

    def global_fa_angle(self):
        grad_weights = []
        grad_weight_fas = []
        for i in range(len(self.classification_layers)):
            grad_weights.append(self.classification_layers[i].grad_weight.flatten())
            grad_weight_fas.append(self.classification_layers[i].grad_weight_fa.flatten())

        grad_weights = torch.cat(grad_weights)
        grad_weight_fas = torch.cat(grad_weight_fas)

        a = (180.0 / math.pi) * (torch.acos(similarity(grad_weights,
                                                       grad_weight_fas))).item()

        return a

    def fa_to_bp_angles(self):
        fa_to_bp_angles = []

        for i in range(len(self.classification_layers)):
            a = (180.0 / math.pi) * (torch.acos(similarity(self.classification_layers[i].grad_weight_bp.flatten(),
                                                           self.classification_layers[
                                                               i].grad_weight_fa.flatten()))).item()
            fa_to_bp_angles.append(a)

        return fa_to_bp_angles

    def global_fa_to_bp_angle(self):
        grad_weight_bps = []
        grad_weight_fas = []
        for i in range(len(self.classification_layers)):
            grad_weight_bps.append(self.classification_layers[i].grad_weight_bp.flatten())
            grad_weight_fas.append(self.classification_layers[i].grad_weight_fa.flatten())

        grad_weight_bps = torch.cat(grad_weight_bps)
        grad_weight_fas = torch.cat(grad_weight_fas)

        a = (180.0 / math.pi) * (torch.acos(similarity(grad_weight_bps,
                                                       grad_weight_fas))).item()

        return a

    def bp_grad_magnitudes(self):
        bp_grad_magnitudes = []
        for i in range(len(self.classification_layers)):
            m = torch.mean(torch.abs(self.classification_layers[i].grad_weight_bp)).item()
            bp_grad_magnitudes.append(m)

        return bp_grad_magnitudes

    def grad_magnitudes(self):
        grad_magnitudes = []
        for i in range(len(self.classification_layers)):
            m = torch.mean(torch.abs(self.classification_layers[i].grad_weight)).item()
            grad_magnitudes.append(m)

        return grad_magnitudes

    def log_layer_states(self):
        # Log hidden
        for i in range(len(self.classification_layers) - 1):
            layer = self.classification_layers[i]
            wandb.log({f"hidden{i}.event_rate": wandb.Histogram(layer.e.flatten().cpu().numpy()),
                       f"hidden{i}.apical": wandb.Histogram(layer.apic.flatten().cpu().numpy()),
                       f"hidden{i}.burst_prob": wandb.Histogram(layer.p_t.flatten().cpu().numpy()),
                       f"hidden{i}.burst_rate": wandb.Histogram(layer.b_t.flatten().cpu().numpy())}, commit=False)

        # Log output
        output_layer = self.classification_layers[-1]
        wandb.log({"output.event_rate": wandb.Histogram(output_layer.e.flatten().cpu().numpy()),
                   "output.burst_prob": wandb.Histogram(output_layer.p_t.flatten().cpu().numpy()),
                   "output.burst_rate": wandb.Histogram(output_layer.b_t.flatten().cpu().numpy())}, commit=False)

    def _initialize_weights(self):
        self._initialize_ff_weights()
        self._initialize_secondary_weights()

    def _initialize_ff_weights(self):
        module_list = list(self.modules())

        for module_index, m in enumerate(module_list):
            if isinstance(m, BurstCCNHiddenLayer) or isinstance(m, BurstCCNOutputLayer) or isinstance(m, Conv2dBurstCCNFinalLayer) or isinstance(m, Conv2dBurstCCNHiddenLayer):
                nn.init.xavier_normal_(m.weight, gain=3.6)
                # nn.init.normal_(m.weight, 0.0, 0.1)
                nn.init.constant_(m.bias, 0)
                # m.weight.data.normal_(0, 0.1)
                # stdv = 1. / math.sqrt(m.weight.size(1))
                # m.bias.data.uniform_(-stdv, stdv)

    def _initialize_secondary_weights(self):
        layer_index = 0
        module_list = list(self.modules())

        for module_index, m in enumerate(module_list):
            if isinstance(m, BurstCCNHiddenLayer) or isinstance(m, Conv2dBurstCCNFinalLayer) or isinstance(m, Conv2dBurstCCNHiddenLayer):
                if self.Y_mode == 'tied' or self.Y_mode == 'symmetric_init':
                    m.weight_Y.data = self.Y_scale * copy.deepcopy(module_list[module_index + 1].weight.detach())
                elif self.Y_mode == 'random_init':
                    init.normal_(m.weight_Y, 0, self.Y_scale)

                if self.Q_mode == 'tied' or self.Q_mode == 'symmetric_init':
                    assert self.p_baseline == 0.5
                    assert self.Q_scale == 1.0
                    m.weight_Q.data = self.p_baseline * copy.deepcopy(m.weight_Y.data)
                elif self.Q_mode == 'W_symmetric_init':
                    m.weight_Q.data = self.p_baseline * self.Q_scale * copy.deepcopy(module_list[module_index + 1].weight.detach())
                elif self.Q_mode == 'random_init':
                    init.normal_(m.weight_Q, 0, self.Q_scale)

                layer_index += 1


class MNISTNetDLBurstCCN(nn.Module):
    def __init__(self, p_baseline, Y_mode, Q_mode, Y_scale, Q_scale,
                 Y_learning, Q_learning,
                 n_hidden_layers, n_hidden_units, device):
        super(MNISTNetDLBurstCCN, self).__init__()

        self.Y_mode = Y_mode
        self.Y_scale = Y_scale
        self.Q_mode = Q_mode
        self.Q_scale = Q_scale

        self.Y_learning = Y_learning
        self.Q_learning = Q_learning

        self.feature_layers = []

        self.feature_layers.append(Flatten())

        self.classification_layers = []

        if n_hidden_layers == 0:
            self.classification_layers.append(
                BurstCCNOutputLayer(784, 10, p_baseline, Y_learning, Q_learning, device))
        elif n_hidden_layers == 1:
            self.classification_layers.append(
                DLBurstCCNHiddenLayer(784, n_hidden_units, 10, p_baseline, Y_learning, Q_learning,
                                      device))
            self.classification_layers.append(
                BurstCCNOutputLayer(n_hidden_units, 10, p_baseline, Y_learning, Q_learning, device))
        else:
            self.classification_layers.append(
                DLBurstCCNHiddenLayer(784, n_hidden_units, n_hidden_units, p_baseline, Y_learning,
                                      Q_learning, device))

            for i in range(1, n_hidden_layers - 1):
                self.classification_layers.append(
                    DLBurstCCNHiddenLayer(n_hidden_units, n_hidden_units, n_hidden_units, p_baseline, Y_learning,
                                          Q_learning, device))

            self.classification_layers.append(
                DLBurstCCNHiddenLayer(n_hidden_units, n_hidden_units, 10, p_baseline, Y_learning,
                                      Q_learning, device))
            self.classification_layers.append(
                BurstCCNOutputLayer(n_hidden_units, 10, p_baseline, Y_learning, Q_learning, device))

        self.out = nn.Sequential(*(self.feature_layers + self.classification_layers))

        self._initialize_weights()

    def set_forward_noise(self, forward_noise):
        for layer in self.classification_layers:
            layer.forward_noise = forward_noise

    def forward(self, x):
        return self.out(x)

    def backward(self, target):
        burst_rate, event_rate, feedback_bp, _ = self.classification_layers[-1].backward(target)
        for i in range(len(self.classification_layers) - 2, -1, -1):
            burst_rate, event_rate, feedback_bp = self.classification_layers[i].backward(burst_rate, event_rate,
                                                                                         feedback_bp)

    def update_weights(self, lr, lr_Y, lr_Q, momentum=0, weight_decay=0, batch_size=1):
        for i in range(len(self.classification_layers)):
            self.classification_layers[i].update_weights(lr=lr[i], lr_Y=lr_Y[i], lr_Q=lr_Q[i], momentum=momentum,
                                                         weight_decay=weight_decay,
                                                         batch_size=batch_size)

    def loss(self, output, target):
        return F.mse_loss(output, target)

    def weight_angles_W_Y(self):
        raise NotImplementedError()
        weight_angles = []

        for i in range(1, len(self.classification_layers)):
            weight_angles.append((180 / math.pi) * torch.acos(
                similarity(self.classification_layers[i].weight.flatten(),
                           (self.classification_layers[i - 1].weight_Y_direct - self.classification_layers[
                               i - 1].weight_Y_from_SST).flatten())).item())

        return weight_angles

    def weight_angles_Q_Y(self):
        raise NotImplementedError()
        weight_angles = []

        for i in range(1, len(self.classification_layers) - 1):
            weight_angles.append((180.0 / math.pi) * torch.acos(
                similarity(self.classification_layers[i].weight_Y.flatten(),
                           self.classification_layers[i].weight_Q.flatten())).item())

        return weight_angles

    def bp_angles(self):
        bp_angles = []

        for i in range(len(self.classification_layers)):
            # a = np.mean([(180.0 / math.pi) * (torch.acos(
            #     similarity(self.classification_layers[i].delta[j].flatten(),
            #                self.classification_layers[i].delta_bp[j].flatten()))).cpu()
            #              for j in range(self.classification_layers[i].delta.shape[0])])

            a = (180.0 / math.pi) * (torch.acos(similarity(self.classification_layers[i].grad_weight.flatten(),
                                                           self.classification_layers[
                                                               i].grad_weight_bp.flatten()))).item()

            if np.isnan(a):
                a = 90.0
                assert (self.classification_layers[i].grad_weight == 0.0).all() or torch.any(
                    torch.isnan(self.classification_layers[i].grad_weight))
                import warnings

                if (self.classification_layers[i].grad_weight == 0.0).all():
                    warnings.warn(f'Updates in layer {i} are 0!')
                elif torch.any(torch.isnan(self.classification_layers[i].grad_weight)):
                    warnings.warn(f'Gradients in layer {i} are NaN!')

            bp_angles.append(a)

        return bp_angles

    def global_bp_angle(self):
        grad_weights = []
        grad_weight_bps = []
        for i in range(len(self.classification_layers)):
            grad_weights.append(self.classification_layers[i].grad_weight.flatten())
            grad_weight_bps.append(self.classification_layers[i].grad_weight_bp.flatten())

        grad_weights = torch.cat(grad_weights)
        grad_weight_bps = torch.cat(grad_weight_bps)

        a = (180.0 / math.pi) * (torch.acos(similarity(grad_weights,
                                                       grad_weight_bps))).item()

        return a

    def SST2_bias_angles(self):
        pass

    def bp_grad_magnitudes(self):
        bp_grad_magnitudes = []
        for i in range(len(self.classification_layers)):
            m = torch.mean(torch.abs(self.classification_layers[i].grad_weight_bp)).item()
            bp_grad_magnitudes.append(m)

        return bp_grad_magnitudes

    def grad_magnitudes(self):
        grad_magnitudes = []
        for i in range(len(self.classification_layers)):
            m = torch.mean(torch.abs(self.classification_layers[i].grad_weight)).item()
            grad_magnitudes.append(m)

        return grad_magnitudes

    def log_layer_states(self):
        # Log hidden
        for i in range(1, len(self.classification_layers) - 1):
            layer = self.classification_layers[i]
            wandb.log({f"hidden{i}.event_rate": wandb.Histogram(layer.e.flatten().cpu().numpy()),
                       f"hidden{i}.apical": wandb.Histogram(layer.apic.flatten().cpu().numpy()),
                       f"hidden{i}.burst_prob": wandb.Histogram(layer.p_t.flatten().cpu().numpy()),
                       f"hidden{i}.burst_rate": wandb.Histogram(layer.b_t.flatten().cpu().numpy())}, commit=False)

        # Log output
        output_layer = self.classification_layers[-1]
        wandb.log({"output.event_rate": wandb.Histogram(output_layer.e.flatten().cpu().numpy()),
                   "output.burst_prob": wandb.Histogram(output_layer.p_t.flatten().cpu().numpy()),
                   "output.burst_rate": wandb.Histogram(output_layer.b_t.flatten().cpu().numpy())}, commit=False)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, DLBurstCCNHiddenLayer) or isinstance(m, BurstCCNOutputLayer):
                nn.init.xavier_normal_(m.weight, gain=3.6)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, DLBurstCCNHiddenLayer):
                # init.constant_(m.weight_Y_direct, 0.0)
                init.constant_(m.weight_Y_from_SST2, 0.0)
                init.constant_(m.weight_Y_from_SST, 0.0)

                effective_Y_weights = torch.randn(m.weight_Y_from_SST.shape,
                                                  device=m.weight_Y_from_SST.device) * self.Y_scale

                m.weight_Y_from_SST2[effective_Y_weights >= 0.0] = effective_Y_weights[effective_Y_weights >= 0.0]
                m.weight_Y_from_SST[effective_Y_weights <= 0.0] = -effective_Y_weights[effective_Y_weights <= 0.0]

                m.weight_Y_from_SST2 += 2.0 * self.Y_scale
                m.weight_Y_from_SST += 2.0 * self.Y_scale

                m.weight_Y_to_SST = torch.eye(m.weight_Y_to_SST.shape[0], device=m.weight_Y_to_SST.device)
                # m.weight_Y_to_SST = generate_positive_full_rank_matrix(m.weight_Y_to_SST.shape[0], device=m.weight_Y_to_SST.device)

                m.weight_Y_to_VIP = torch.eye(m.weight_Y_to_VIP.shape[0], device=m.weight_Y_to_VIP.device)
                m.weight_Y_VIP_to_SST = torch.eye(m.weight_Y_VIP_to_SST.shape[0], device=m.weight_Y_VIP_to_SST.device)

                # m.weight_Y_to_VIP = generate_positive_full_rank_matrix(m.weight_Y_to_VIP.shape[0], device=m.weight_Y_to_VIP.device)
                # m.weight_Y_VIP_to_SST = generate_positive_full_rank_matrix(m.weight_Y_VIP_to_SST.shape[0], device=m.weight_Y_VIP_to_SST.device)

                init.constant_(m.weight_Q_direct, 0.0)
                init.constant_(m.weight_Q_from_NDNF, 0.0)

                # effective_Q_weights = torch.randn(m.weight_Q_from_NDNF.shape, device=m.weight_Q_from_NDNF.device) * self.Q_scale
                #
                # m.weight_Q_direct[effective_Q_weights >= 0.0] = effective_Q_weights[effective_Q_weights >= 0.0]
                # m.weight_Q_from_NDNF[effective_Q_weights <= 0.0] = -effective_Q_weights[effective_Q_weights <= 0.0]
                #
                # m.weight_Q_direct += 2.0 * self.Q_scale
                # m.weight_Q_from_NDNF += 2.0 * self.Q_scale

                m.weight_Q_to_NDNF = torch.eye(m.weight_Q_to_NDNF.shape[0], device=m.weight_Q_to_NDNF.device)
                # m.weight_Q_to_NDNF = generate_positive_full_rank_matrix(m.weight_Q_to_NDNF.shape[0], device=m.weight_Q_to_NDNF.device)

                m.weight_Q_direct = 0.5 * copy.deepcopy(m.weight_Y_from_SST)
                m.weight_Q_from_NDNF = 0.5 * copy.deepcopy(m.weight_Y_from_SST2)
                # m.weight_Q_to_NDNF = torch.eye(m.weight_Q_to_NDNF.shape[0], device=m.weight_Q_to_NDNF.device)

    def _initialize_weights_from_list(self, ff_weights_list, ff_bias_list):
        layer_index = 0
        for m in self.modules():
            if isinstance(m, DLBurstCCNHiddenLayer) or isinstance(m, BurstCCNOutputLayer):
                m.weight = copy.deepcopy(ff_weights_list[layer_index].detach())
                m.bias = copy.deepcopy(ff_bias_list[layer_index].detach())

            if isinstance(m, DLBurstCCNHiddenLayer):
                init.constant_(m.weight_Y_from_SST2, 0.0)
                init.constant_(m.weight_Y_from_SST, 0.0)
                # init.normal_(m.weight_Y, 0, self.weight_Y_std)
                effective_Y_weights = copy.deepcopy(ff_weights_list[layer_index + 1].detach())

                # m.weight_Y_to_SST = torch.eye(m.weight_Y_to_SST.shape[0], device=m.weight_Y_to_SST.device)
                m.weight_Y_to_SST = generate_positive_full_rank_matrix(m.weight_Y_to_SST.shape[0],
                                                                       device=m.weight_Y_to_SST.device)

                # m.weight_Y_to_VIP = torch.eye(m.weight_Y_to_VIP.shape[0], device=m.weight_Y_to_VIP.device)
                # m.weight_Y_VIP_to_SST = torch.eye(m.weight_Y_VIP_to_SST.shape[0], device=m.weight_Y_VIP_to_SST.device)

                m.weight_Y_to_VIP = generate_positive_full_rank_matrix(m.weight_Y_to_VIP.shape[0],
                                                                       device=m.weight_Y_to_VIP.device)
                m.weight_Y_VIP_to_SST = generate_positive_full_rank_matrix(m.weight_Y_VIP_to_SST.shape[0],
                                                                           device=m.weight_Y_VIP_to_SST.device)

                m.weight_Y_from_SST2[effective_Y_weights >= 0.0] = copy.deepcopy(
                    effective_Y_weights[effective_Y_weights >= 0.0])
                m.weight_Y_from_SST[effective_Y_weights <= 0.0] = -copy.deepcopy(
                    effective_Y_weights[effective_Y_weights <= 0.0])

                m.weight_Y_from_SST2 += 2.0 * self.weight_Y_std
                m.weight_Y_from_SST += 2.0 * self.weight_Y_std

                # m.weight_Y_from_SST2 = torch.inverse(m.weight_Y_to_VIP.mm(m.weight_Y_VIP_to_SST)).mm(m.weight_Y_from_SST2)
                m.weight_Y_from_SST2 = torch.inverse(m.weight_Y_to_VIP.mm(m.weight_Y_VIP_to_SST)).mm(
                    m.weight_Y_from_SST2)
                m.weight_Y_from_SST = torch.inverse(m.weight_Y_to_SST).mm(m.weight_Y_from_SST)

                # m.weight_Q_direct = 0.5 * copy.deepcopy(m.weight_Y_from_SST)
                # m.weight_Q_from_NDNF = 0.5 * copy.deepcopy(m.weight_Y_from_SST2)
                # m.weight_Q_to_NDNF = torch.eye(m.weight_Q_to_NDNF.shape[0], device=m.weight_Q_to_NDNF.device)

            if isinstance(m, DLBurstCCNHiddenLayer) or isinstance(m, BurstCCNOutputLayer):
                layer_index += 1
