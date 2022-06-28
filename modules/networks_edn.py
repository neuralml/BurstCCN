import copy

import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from modules.layers import Flatten
from modules.layers_edn import EDNOutputLayer, EDNHiddenLayer

import wandb

from helpers import similarity
from modules.optimisers import NetworkCostOptimiser


class MNISTNetEDN(nn.Module):
    def __init__(self, n_hidden_layers, n_hidden_units, lambda_output, lambda_intn, lambda_hidden, Y_mode, Y_scale, Y_learning, intn_feedback_mode, intn_feedback_scale, device):
        super(MNISTNetEDN, self).__init__()

        # self.weight_scales = weight_scales
        # self.weight_Y_scales = weight_Y_scales
        # self.pyr_intn_weight_scales = pyr_intn_weight_scales
        # self.intn_pyr_weight_scales = intn_pyr_weight_scales

        assert Y_mode in ['tied', 'symmetric_init', 'random_init']
        self.Y_mode = Y_mode
        self.Y_scale = Y_scale
        self.Y_learning = Y_learning
        self.intn_feedback_mode = intn_feedback_mode
        self.intn_feedback_scale = intn_feedback_scale

        self.device = device

        self.feature_layers = []

        self.feature_layers.append(Flatten())

        self.classification_layers = []

        if n_hidden_layers == 0:
            self.classification_layers.append(
                EDNOutputLayer(784, 10, lambda_output, device))
        elif n_hidden_layers == 1:
            self.classification_layers.append(
                EDNHiddenLayer(784, n_hidden_units, 10, lambda_intn, lambda_hidden, Y_learning, device))
            self.classification_layers.append(
                EDNOutputLayer(n_hidden_units, 10, lambda_output, device))
        else:
            self.classification_layers.append(
                EDNHiddenLayer(784, n_hidden_units, n_hidden_units, lambda_intn, lambda_hidden, Y_learning, device))

            for i in range(1, n_hidden_layers - 1):
                self.classification_layers.append(
                    EDNHiddenLayer(n_hidden_units, n_hidden_units, n_hidden_units, lambda_intn, lambda_hidden, Y_learning, device))

            self.classification_layers.append(
                EDNHiddenLayer(n_hidden_units, n_hidden_units, 10, lambda_intn, lambda_hidden, Y_learning, device))
            self.classification_layers.append(
                EDNOutputLayer(n_hidden_units, 10, lambda_output, device))

        self.out = nn.Sequential(*(self.feature_layers + self.classification_layers))

        self._initialize_weights()

    def forward(self, x):
        return self.out(x)

    def backward(self, target):
        pyr_soma_t, pyr_soma_rate_t, bp_grad, fa_delta = self.classification_layers[-1].backward(target)
        for i in range(len(self.classification_layers) - 2, -1, -1):
            pyr_soma_t, pyr_soma_rate_t, bp_grad, fa_delta = self.classification_layers[i].backward(pyr_soma_t, pyr_soma_rate_t, bp_grad, fa_delta)

    def update_weights(self, lr, lr_Y,  lr_pyr_intn, lr_intn_pyr, weight_decay=0, optimiser=None, global_cost=None):
        weight_grads = [self.classification_layers[i].grad_weight.detach() for i in range(len(self.classification_layers))]
        bias_grads = [self.classification_layers[i].grad_bias.detach() for i in range(len(self.classification_layers))]

        if isinstance(optimiser, NetworkCostOptimiser):
            optimiser.update_buffers(global_cost=global_cost)
        else:
            optimiser.update_buffers(weight_grads=weight_grads, bias_grads=bias_grads)

        weight_updates, bias_updates = optimiser.compute_updates(lrs=lr, weight_grads=weight_grads, bias_grads=bias_grads)

        for i in range(len(self.classification_layers)):
            self.classification_layers[i].update_weights(weight_updates[i], bias_updates[i], weight_decay=weight_decay)
            self.classification_layers[i].update_secondary_weights(lr_Y=lr_Y[i],  lr_pyr_intn=lr_pyr_intn[i], lr_intn_pyr=lr_intn_pyr[i])

        # for i in range(len(self.classification_layers)):
        #     self.classification_layers[i].update_weights(lr=lrs[i], momentum=momentum,
        #                                                  weight_decay=weight_decay,
        #                                                  batch_size=batch_size)
        for i in range(len(self.classification_layers)):
            if i != len(self.classification_layers) - 1 and self.Y_mode == 'tied':
                self.classification_layers[i].weight_Y = self.Y_scale * copy.deepcopy(self.classification_layers[i + 1].weight)

            if i != len(self.classification_layers) - 1 and self.intn_feedback_mode == 'tied':
                self.classification_layers[i].weight_pyr_intn = copy.deepcopy(self.classification_layers[i + 1].weight.detach())
                self.classification_layers[i].bias_pyr_intn = copy.deepcopy(self.classification_layers[i + 1].bias.detach())
                self.classification_layers[i].weight_intn_pyr = self.intn_feedback_scale * copy.deepcopy(self.classification_layers[i].weight_Y.detach())

    # def update_weights(self, lr, lr_Y,  lr_pyr_intn, lr_intn_pyr, momentum=0, weight_decay=0, batch_size=1):
    #     for i in range(len(self.classification_layers)):
    #         self.classification_layers[i].update_weights(lr_ff=lr[i], lr_Y=lr_Y[i],  lr_pyr_intn=lr_pyr_intn[i], lr_intn_pyr=lr_intn_pyr[i], momentum=momentum, weight_decay=weight_decay, batch_size=batch_size)
    #
    #     for i in range(len(self.classification_layers)):
    #         if i != len(self.classification_layers) - 1 and self.Y_mode == 'tied':
    #             self.classification_layers[i].weight_Y = self.Y_scale * self.classification_layers[i + 1].weight

    def loss(self, output, target):
        return F.mse_loss(output, target)

    def weight_angles_W_Y(self):
        weight_angles = []

        for i in range(1, len(self.classification_layers)):
            weight_angles.append((180.0 / math.pi) * torch.acos(
                similarity(self.classification_layers[i].weight.flatten(),
                           self.classification_layers[i - 1].weight_Y.flatten())).item())

        return weight_angles

    def weight_angles_ff(self):
        weight_angles = []

        for i in range(1, len(self.classification_layers)):
            weight_angles.append((180.0 / math.pi) * torch.acos(
                similarity(self.classification_layers[i].weight.flatten(),
                           self.classification_layers[i - 1].weight_pyr_intn.flatten())).item())

        return weight_angles

    def weight_angles_fb(self):
        weight_angles = []

        for i in range(len(self.classification_layers) - 1):
            weight_angles.append((180.0 / math.pi) * torch.acos(
                similarity(self.classification_layers[i].weight_Y.flatten(),
                           self.classification_layers[i].weight_intn_pyr.flatten())).item())

        return weight_angles


    def bp_angles(self):
        bp_angles = []

        for i in range(len(self.classification_layers)):
            # a = np.mean([(180.0 / math.pi) * (torch.acos(
            #     similarity(self.classification_layers[i].delta[j].flatten(),
            #                self.classification_layers[i].delta_bp[j].flatten()))).cpu()
            #              for j in range(self.classification_layers[i].delta.shape[0])])

            a = (180.0 / math.pi) * (torch.acos(similarity(self.classification_layers[i].grad_weight.flatten(),
                                                           self.classification_layers[i].grad_weight_bp.flatten()))).item()

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
                                                           self.classification_layers[i].grad_weight_fa.flatten()))).item()
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
                                                           self.classification_layers[i].grad_weight_fa.flatten()))).item()
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
        for i in range(1, len(self.classification_layers) - 1):
            layer = self.classification_layers[i]
            # wandb.log({f"hidden{i}.event_rate": wandb.Histogram(layer.e.flatten().cpu().numpy()),
            #            f"hidden{i}.apical": wandb.Histogram(layer.apic.flatten().cpu().numpy()),
            #            f"hidden{i}.burst_prob": wandb.Histogram(layer.p_t.flatten().cpu().numpy()),
            #            f"hidden{i}.burst_rate": wandb.Histogram(layer.b_t.flatten().cpu().numpy())}, commit=False)

        # Log output
        output_layer = self.classification_layers[-1]
        # wandb.log({"output.event_rate": wandb.Histogram(output_layer.e.flatten().cpu().numpy()),
        #            "output.burst_prob": wandb.Histogram(output_layer.p_t.flatten().cpu().numpy()),
        #            "output.burst_rate": wandb.Histogram(output_layer.b_t.flatten().cpu().numpy())}, commit=False)

    def _initialize_weights(self):
        self._initialize_ff_weights()
        self._initialize_secondary_weights()

    def _initialize_ff_weights(self):
        layer_index = 0
        module_list = list(self.modules())

        for module_index, m in enumerate(module_list):
            if isinstance(m, EDNHiddenLayer) or isinstance(m, EDNOutputLayer):
                nn.init.xavier_normal_(m.weight, gain=3.6)
                nn.init.constant_(m.bias, 0)
                # nn.init.uniform_(m.weight, -self.weight_scales[layer_index], self.weight_scales[layer_index])
                # nn.init.constant_(m.ff_bias, 0)

                # nn.init.uniform_(m.weight, -0.1, 0.1)
                # nn.init.constant_(m.bias, 0)

                layer_index += 1

    def _initialize_secondary_weights(self):
        layer_index = 0
        module_list = list(self.modules())

        for module_index, m in enumerate(module_list):
            if isinstance(m, EDNHiddenLayer):
                # Random init
                #nn.init.uniform_(m.weight_Y, -self.weight_Y_scales[layer_index], self.weight_Y_scales[layer_index])

                # nn.init.uniform_(m.pyr_intn_weight, -self.pyr_intn_weight_scales[layer_index], self.pyr_intn_weight_scales[layer_index])
                # nn.init.constant_(m.pyr_intn_bias, 0)
                #
                # nn.init.uniform_(m.intn_pyr_weight, -self.intn_pyr_weight_scales[layer_index], self.intn_pyr_weight_scales[layer_index])

                # Tied symmetric
                # m.weight_Y = module_list[module_index + 1].weight
                # m.pyr_intn_weight = copy.deepcopy(module_list[module_index + 1].weight)
                # m.pyr_intn_bias = copy.deepcopy(module_list[module_index + 1].ff_bias)
                # m.intn_pyr_weight = copy.deepcopy(module_list[module_index + 1].weight)
                #

                if self.Y_mode == 'tied' or self.Y_mode == 'symmetric_init':
                    m.weight_Y = self.Y_scale * copy.deepcopy(module_list[module_index + 1].weight.detach())
                    # m.pyr_intn_weight = copy.deepcopy(module_list[module_index + 1].weight.detach())
                    # m.pyr_intn_bias = copy.deepcopy(module_list[module_index + 1].ff_bias.detach())
                    # m.intn_pyr_weight = copy.deepcopy(m.weight_Y.detach())
                elif self.Y_mode == 'random_init':
                    # nn.init.uniform_(m.weight_Y, -self.weight_Y_scales[layer_index], self.weight_Y_scales[layer_index])
                    # nn.init.uniform_(m.weight_Y, -self.Y_scale, self.Y_scale)
                    init.normal_(m.weight_Y, 0, self.Y_scale)
                else:
                    raise ValueError(f"Invalid feedback mode {self.Y_mode}")

                if self.intn_feedback_mode == 'tied' or self.intn_feedback_mode == 'symmetric_init':
                    m.weight_pyr_intn = copy.deepcopy(module_list[module_index + 1].weight.detach())
                    m.bias_pyr_intn = copy.deepcopy(module_list[module_index + 1].bias.detach())
                    m.weight_intn_pyr = self.intn_feedback_scale * copy.deepcopy(m.weight_Y.detach())
                elif self.intn_feedback_mode == 'random_init':
                    nn.init.xavier_normal_(m.weight_pyr_intn, gain=3.6)
                    nn.init.constant_(m.bias_pyr_intn, 0)

                    init.normal_(m.weight_intn_pyr, 0, self.intn_feedback_scale)
                else:
                    raise ValueError(f"Invalid interneuron feedback mode: {self.intn_feedback_mode}")

                # Fb alignment random
                # nn.init.uniform_(m.weight_Y, -self.weight_Y_scales[layer_index], self.weight_Y_scales[layer_index])
                # m.pyr_intn_weight = copy.deepcopy(module_list[module_index + 1].weight)
                # m.pyr_intn_bias = copy.deepcopy(module_list[module_index + 1].ff_bias)
                # m.intn_pyr_weight = copy.deepcopy(m.weight_Y)

                layer_index += 1

    def set_forward_noise(self, forward_noise):
        for layer in self.classification_layers:
            layer.forward_noise = forward_noise

    def _initialize_weights_from_list(self, weights_list, ff_bias_list):
        layer_index = 0
        for m in self.modules():
            if isinstance(m, EDNHiddenLayer) or isinstance(m, EDNOutputLayer):
                m.weight = copy.deepcopy(weights_list[layer_index].detach().type(self.dtype))
                m.bias = copy.deepcopy(ff_bias_list[layer_index].detach().type(self.dtype))
                layer_index += 1