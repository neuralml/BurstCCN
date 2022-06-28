import math
import numpy as np

import copy
from torch.nn import init

from modules.layers import Flatten
from modules.layers_burstprop import *

import wandb

from helpers import similarity


class BurstProp(nn.Module):
    def __init__(self, p_baseline, n_hidden_layers, n_hidden_units, recurrent_input, Y_learning, Y_mode, Y_scale, recurrent_learning, recurrent_scale, device):
        super(BurstProp, self).__init__()

        self.recurrent_input = recurrent_input
        self.Y_learning = Y_learning

        assert Y_mode in ['tied', 'symmetric_init', 'random_init']
        self.Y_mode = Y_mode
        self.Y_scale = Y_scale

        self.recurrent_learning = recurrent_learning
        self.recurrent_scale = recurrent_scale

        self.device = device

        self.feature_layers = []
        self.feature_layers.append(Flatten())

        self.classification_layers = []

        if n_hidden_layers == 0:
            self.classification_layers.append(BurstPropOutputLayer(784, 10, p_baseline, Y_learning, device))
        else:
            self.classification_layers.append(
                BurstPropHiddenLayer(784, n_hidden_units, p_baseline, Y_learning, recurrent_input, recurrent_learning, device))

            for i in range(1, n_hidden_layers):
                self.classification_layers.append(
                    BurstPropHiddenLayer(n_hidden_units, n_hidden_units, p_baseline, Y_learning, recurrent_input, recurrent_learning, device))

            self.classification_layers.append(BurstPropOutputLayer(n_hidden_units, 10, p_baseline, Y_learning, device))

        self.out = nn.Sequential(*(self.feature_layers + self.classification_layers))

        self._initialize_weights()

    def set_forward_noise(self, forward_noise):
        for layer in self.classification_layers:
            layer.forward_noise = forward_noise

    def forward(self, x):
        return self.out(x)

    def backward(self, target):
        feedback, feedback_t, feedback_bp, feedback_fa = self.classification_layers[-1].backward(target)
        for i in range(len(self.classification_layers) - 2, -1, -1):
            if self.recurrent_input:
                self.classification_layers[i].backward_pre(feedback)
            feedback, feedback_t, feedback_bp, feedback_fa = self.classification_layers[i].backward(feedback, feedback_t,
                                                                                       feedback_bp, feedback_fa)

    def update_weights(self, lr, momentum=0, weight_decay=0, recurrent_lr=None, batch_size=1):
        for i in range(len(self.classification_layers)):
            if i < len(self.classification_layers) - 1:
                self.classification_layers[i].update_weights(lr=lr[i], momentum=momentum, weight_decay=weight_decay,
                                                             recurrent_lr=recurrent_lr, batch_size=batch_size)

            else:
                self.classification_layers[i].update_weights(lr=lr[i], momentum=momentum, weight_decay=weight_decay,
                                                             batch_size=batch_size)

        for i in range(len(self.classification_layers)):
            if self.Y_mode == 'tied':
                self.classification_layers[i].weight_Y = self.Y_scale * copy.deepcopy(self.classification_layers[i].weight)


    def loss(self, output, target):
        return F.mse_loss(output, target)

    def weight_angles_W_Y(self):
        weight_angles_W_Y = []

        for i in range(1, len(self.classification_layers)):
            # weight_angles_W_Y.append((180 / math.pi) * torch.acos(
            #     F.cosine_similarity(self.classification_layers[i].weight.flatten(),
            #                         self.classification_layers[i].weight_Y.flatten(), dim=0).clip(-1.0, 1.0)))

            weight_angles_W_Y.append((180 / math.pi) * torch.acos(
                similarity(self.classification_layers[i].weight.flatten(),
                           self.classification_layers[i].weight_Y.flatten())).item())


        return weight_angles_W_Y

    def bp_angles(self):
        bp_angles = []

        for i in range(len(self.classification_layers)):
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

    def _initialize_weights(self):
        self._initialize_ff_weights()
        self._initialize_secondary_weights()

    def _initialize_ff_weights(self):
        layer_index = 0
        module_list = list(self.modules())

        for module_index, m in enumerate(module_list):
            if isinstance(m, BurstPropHiddenLayer) or isinstance(m, BurstPropOutputLayer):
                nn.init.xavier_normal_(m.weight, gain=3.6)
                nn.init.constant_(m.bias, 0)

                layer_index += 1

    def _initialize_secondary_weights(self):
        layer_index = 0
        module_list = list(self.modules())

        for module_index, m in enumerate(module_list):
            if isinstance(m, BurstPropHiddenLayer) or isinstance(m, BurstPropOutputLayer):
                if self.Y_mode == 'tied' or self.Y_mode == 'symmetric_init':
                    m.weight_Y = self.Y_scale * copy.deepcopy(module_list[module_index].weight.detach())
                elif self.Y_mode == 'random_init':
                    init.normal_(m.weight_Y, 0, self.Y_scale)

                # init.normal_(m.weight_Y, 0, self.Y_scale)

            if self.recurrent_input and isinstance(m, BurstPropHiddenLayer):
                init.normal_(m.weight_r, 0, self.recurrent_scale)


    def _initialize_weights_from_list(self, ff_weights_list, ff_bias_list):
        raise NotImplementedError
        layer_index = 0
        for m in self.modules():
            if isinstance(m, BurstPropHiddenLayer) or isinstance(m, BurstPropOutputLayer):
                m.weight = copy.deepcopy(ff_weights_list[layer_index].detach())
                m.bias = copy.deepcopy(ff_bias_list[layer_index].detach())

                m.weight_Y = copy.deepcopy(ff_weights_list[layer_index].detach())

                if self.recurrent_input and isinstance(m, BurstPropHiddenLayer):
                    init.normal_(m.weight_r, 0, self.recurrent_scale)
                #     m.weight_r = recurrent_weights_list[layer_index]

                layer_index += 1


class ConvBurstProp(nn.Module):
    def __init__(self, input_channels, p_baseline, Y_mode, Y_scale, recurrent_scale, Y_learning, recurrent_input,
                 recurrent_learning, device):
        super(ConvBurstProp, self).__init__()

        self.Y_mode = Y_mode
        self.Y_scale = Y_scale
        self.recurrent_scale = recurrent_scale
        self.Y_learning = Y_learning
        self.recurrent_input = recurrent_input
        self.recurrent_learning = recurrent_learning

        self.device = device

        self.feature_layers = []

        if self.Y_learning:
            self.feature_layers.append(
                Conv2dBurstPropHiddenLayer(input_channels, 64, p_baseline, Y_learning, False, False, 32, device,
                                           kernel_size=5, stride=2))
            self.feature_layers.append(Conv2dBurstPropHiddenLayer(64, 128, p_baseline, Y_learning, False, False,
                                                                  self.feature_layers[0].out_size, device, kernel_size=5,
                                                                  stride=2))
            self.feature_layers.append(Conv2dBurstPropHiddenLayer(128, 256, p_baseline, Y_learning, False, False,
                                                                  self.feature_layers[1].out_size, device, kernel_size=3))
        else:
            self.feature_layers.append(
                Conv2dBurstPropHiddenLayer(input_channels, 64, p_baseline, Y_learning, False, False, 32, device,
                                           kernel_size=5, stride=2))
            self.feature_layers.append(Conv2dBurstPropHiddenLayer(64, 256, p_baseline, Y_learning, False, False,
                                                                  self.feature_layers[0].out_size, device, kernel_size=5,
                                                                  stride=2))
            self.feature_layers.append(Conv2dBurstPropHiddenLayer(256, 256, p_baseline, Y_learning, False, False,
                                                                  self.feature_layers[1].out_size, device, kernel_size=3))
        self.feature_layers.append(Flatten())

        self.classification_layers = []

        if self.Y_learning:
            self.classification_layers.append(
                BurstPropHiddenLayer(2304, 1024, p_baseline, Y_learning, recurrent_input, recurrent_learning, device))
            self.classification_layers.append(BurstPropOutputLayer(1024, 10, p_baseline, Y_learning, device))
        else:
            self.classification_layers.append(
                BurstPropHiddenLayer(2304, 1480, p_baseline, Y_learning, recurrent_input, recurrent_learning, device))
            self.classification_layers.append(BurstPropOutputLayer(1480, 10, p_baseline, Y_learning, device))

        self.out = nn.Sequential(*(self.feature_layers + self.classification_layers))

        self._initialize_weights()

    def forward(self, x):
        return self.out(x)

    def backward(self, target):
        feedback, feedback_t, feedback_bp, feedback_fa = self.classification_layers[-1].backward(target)
        for i in range(len(self.classification_layers) - 2, -1, -1):
            if self.recurrent_input:
                self.classification_layers[i].backward_pre(feedback)
            feedback, feedback_t, feedback_bp, feedback_fa = self.classification_layers[i].backward(feedback, feedback_t,
                                                                                       feedback_bp, feedback_fa)

        for i in range(len(self.feature_layers) - 1, -1, -1):
            if self.recurrent_input and i < len(self.feature_layers) - 1:
                self.feature_layers[i].backward_pre(feedback)
            feedback, feedback_t, feedback_bp, feedback_fa = self.feature_layers[i].backward(feedback, feedback_t, feedback_bp, feedback_fa)

    def update_weights(self, lr, momentum=0, weight_decay=0, recurrent_lr=None, batch_size=1):
        for i in range(len(self.classification_layers)):
            if i < len(self.classification_layers) - 1:
                self.classification_layers[i].update_weights(lr=lr[len(self.feature_layers) - 1 + i], momentum=momentum,
                                                             weight_decay=weight_decay, recurrent_lr=recurrent_lr, batch_size=batch_size)
            else:
                self.classification_layers[i].update_weights(lr=lr[len(self.feature_layers) - 1 + i], momentum=momentum,
                                                             weight_decay=weight_decay, batch_size=batch_size)

        for i in range(len(self.feature_layers) - 1):
            self.feature_layers[i].update_weights(lr=lr[i], momentum=momentum, weight_decay=weight_decay,
                                                  recurrent_lr=recurrent_lr, batch_size=batch_size)
        if self.Y_mode == 'tied':
            for i in range(len(self.classification_layers)):
                self.classification_layers[i].weight_Y = self.Y_scale * copy.deepcopy(self.classification_layers[i].weight)

            for i in range(len(self.feature_layers) - 1):
                self.feature_layers[i].weight_Y = self.Y_scale * copy.deepcopy(self.feature_layers[i].weight)

    def loss(self, output, target):
        return F.mse_loss(output, target)

    def weight_angles_W_Y(self):
        weight_angles = []

        for i in range(1, len(self.feature_layers) - 1):
            weight_angles.append((180 / math.pi) * torch.acos(
                F.cosine_similarity(self.feature_layers[i].weight.flatten(), self.feature_layers[i].weight_Y.flatten(),
                                    dim=0)))

        for i in range(len(self.classification_layers)):
            weight_angles.append((180 / math.pi) * torch.acos(
                F.cosine_similarity(self.classification_layers[i].weight.flatten(),
                                    self.classification_layers[i].weight_Y.flatten(), dim=0)))

        return weight_angles

    def bp_angles(self):
        bp_angles = []

        for i in range(len(self.feature_layers) - 1):
            a = (180.0 / math.pi) * (torch.acos(similarity(self.feature_layers[i].grad_weight.flatten(),
                                                           self.feature_layers[
                                                               i].grad_weight_bp.flatten()))).item()

            if np.isnan(a):
                a = 90.0
                assert (self.feature_layers[i].grad_weight == 0.0).all() or torch.any(
                    torch.isnan(self.feature_layers[i].grad_weight))
                import warnings

                if (self.feature_layers[i].grad_weight == 0.0).all():
                    warnings.warn(f'Updates in layer {i} are 0!')
                elif torch.any(torch.isnan(self.feature_layers[i].grad_weight)):
                    warnings.warn(f'Gradients in layer {i} are NaN!')

            bp_angles.append(a)

        for i in range(len(self.classification_layers)):
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

        for i in range(len(self.feature_layers) - 1):
            grad_weights.append(self.feature_layers[i].grad_weight.flatten())
            grad_weight_bps.append(self.feature_layers[i].grad_weight_bp.flatten())


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

        for i in range(len(self.feature_layers) - 1):
            a = (180.0 / math.pi) * (torch.acos(similarity(self.feature_layers[i].grad_weight.flatten(),
                                                           self.feature_layers[
                                                               i].grad_weight_fa.flatten()))).item()
            fa_angles.append(a)

        for i in range(len(self.classification_layers)):
            a = (180.0 / math.pi) * (torch.acos(similarity(self.classification_layers[i].grad_weight.flatten(),
                                                           self.classification_layers[
                                                               i].grad_weight_fa.flatten()))).item()
            fa_angles.append(a)

        return fa_angles

    def global_fa_angle(self):
        grad_weights = []
        grad_weight_fas = []

        for i in range(len(self.feature_layers) - 1):
            grad_weights.append(self.feature_layers[i].grad_weight.flatten())
            grad_weight_fas.append(self.feature_layers[i].grad_weight_fa.flatten())

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

        for i in range(len(self.feature_layers) - 1):
            a = (180.0 / math.pi) * (torch.acos(similarity(self.feature_layers[i].grad_weight_bp.flatten(),
                                                           self.feature_layers[
                                                               i].grad_weight_fa.flatten()))).item()
            fa_to_bp_angles.append(a)

        for i in range(len(self.classification_layers)):
            a = (180.0 / math.pi) * (torch.acos(similarity(self.classification_layers[i].grad_weight_bp.flatten(),
                                                           self.classification_layers[
                                                               i].grad_weight_fa.flatten()))).item()
            fa_to_bp_angles.append(a)

        return fa_to_bp_angles

    def global_fa_to_bp_angle(self):
        grad_weight_bps = []
        grad_weight_fas = []

        for i in range(len(self.feature_layers) - 1):
            grad_weight_bps.append(self.feature_layers[i].grad_weight_bp.flatten())
            grad_weight_fas.append(self.feature_layers[i].grad_weight_fa.flatten())

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

        for i in range(len(self.feature_layers) - 1):
            m = torch.mean(torch.abs(self.feature_layers[i].grad_weight_bp)).item()
            bp_grad_magnitudes.append(m)

        for i in range(len(self.classification_layers)):
            m = torch.mean(torch.abs(self.classification_layers[i].grad_weight_bp)).item()
            bp_grad_magnitudes.append(m)

        return bp_grad_magnitudes

    def grad_magnitudes(self):
        grad_magnitudes = []

        for i in range(len(self.feature_layers) - 1):
            m = torch.mean(torch.abs(self.feature_layers[i].grad_weight)).item()
            grad_magnitudes.append(m)

        for i in range(len(self.classification_layers)):
            m = torch.mean(torch.abs(self.classification_layers[i].grad_weight)).item()
            grad_magnitudes.append(m)

        return grad_magnitudes

    def _initialize_weights(self):
        self._initialize_ff_weights()
        self._initialize_secondary_weights()

    def _initialize_ff_weights(self):
        module_list = list(self.modules())

        for module_index, m in enumerate(module_list):
            if isinstance(m, Conv2dBurstPropHiddenLayer) or isinstance(m, BurstPropHiddenLayer) or isinstance(m, BurstPropOutputLayer):
                nn.init.xavier_normal_(m.weight, gain=3.6)
                nn.init.constant_(m.bias, 0)

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, Conv2dHiddenLayer) or isinstance(m, BurstPropHiddenLayer) or isinstance(m, BurstPropOutputLayer):
    #             nn.init.xavier_normal_(m.weight, gain=3.6)
    #             nn.init.constant_(m.bias, 0)
    #
    #             init.normal_(m.weight_Y, 0, self.Y_scale)
    #
    #             if self.recurrent_input and isinstance(m, BurstPropHiddenLayer):
    #                 init.uniform_(m.weight_r, -self.recurrent_scale, self.recurrent_scale)
    def _initialize_secondary_weights(self):
        layer_index = 0
        module_list = list(self.modules())

        for module_index, m in enumerate(module_list):
            if isinstance(m, Conv2dBurstPropHiddenLayer) or isinstance(m, BurstPropHiddenLayer) or isinstance(m, BurstPropOutputLayer):
                if self.Y_mode == 'tied' or self.Y_mode == 'symmetric_init':
                    m.weight_Y.data = self.Y_scale * copy.deepcopy(module_list[module_index].weight.detach())
                elif self.Y_mode == 'random_init':
                    init.normal_(m.weight_Y, 0, self.Y_scale)

                if self.recurrent_input and isinstance(m, BurstPropHiddenLayer):
                    init.uniform_(m.weight_r, -self.recurrent_scale, self.recurrent_scale)

                layer_index += 1