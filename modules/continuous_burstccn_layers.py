import math

import torch


class ContinuousBurstCCNOutputLayer:
    def __init__(self, in_features, out_features, p_baseline, device):
        self.in_features = in_features
        self.out_features = out_features
        self.p_baseline = p_baseline
        self.device = device

        self.soma_potential = torch.zeros((out_features, 1), device=device)
        self.soma_potential_cache = torch.zeros((out_features, 1), device=device)

        self.event_rate = torch.zeros((out_features, 1), device=device)
        self.event_rate_cache = torch.zeros((out_features, 1), device=device)

        self.event_rate_ma = torch.zeros((out_features, 1), device=device)
        self.event_rate_ma_cache = torch.zeros((out_features, 1), device=device)

        self.burst_rate = torch.zeros((out_features, 1), device=device)
        self.burst_rate_cache = torch.zeros((out_features, 1), device=device)

        self.burst_rate_ma = torch.zeros((out_features, 1), device=device)
        self.burst_rate_ma_cache = torch.zeros((out_features, 1), device=device)

        self.dendritic_potential = torch.zeros((out_features, 1), device=device)
        self.dendritic_potential_cache = torch.zeros((out_features, 1), device=device)

        self.burst_prob = torch.zeros((out_features, 1), device=device)
        self.burst_prob_cache = torch.zeros((out_features, 1), device=device)

        # self.burst_prob_ma = torch.zeros((out_features, 1), device=device)
        # self.burst_prob_ma_cache = torch.zeros((out_features, 1), device=device)

        self.beta = 1
        self.lambda_ = 0.05
        self.tau_u = 0.1
        self.tau_z = 0.1
        self.tau_e = self.tau_z
        self.tau_W = 100.0
        self.tau_moving_avg = 5.0
        self.e_max = 5.0

        self.weight = math.sqrt(2.0 / (in_features + out_features)) * torch.randn(out_features, in_features, device=device)
        self.bias = torch.zeros((out_features, 1), device=device)

        self.small_vector = torch.ones((out_features, 1), device=device) * 1e-8
        self.p_baseline = torch.ones((out_features, 1), device=device) * 0.5

        self.activation_function = torch.sigmoid

        def h_for_sigmoid(x):
            return 1.0 - x

        self.h = h_for_sigmoid

    def cache_state(self):
        self.event_rate_cache = self.event_rate.detach().clone()
        self.burst_rate_cache = self.burst_rate.detach().clone()
        self.burst_prob_cache = self.burst_prob.detach().clone()

        self.event_rate_ma_cache = self.event_rate_ma.detach().clone()
        self.burst_rate_ma_cache = self.burst_rate_ma.detach().clone()
        # self.burst_prob_ma_cache = self.burst_prob_ma.detach().clone()

    def feedforward_update(self, input_event_rate, dt=0.1):
        self.soma_potential = (1. - dt / self.tau_z) * self.soma_potential + (dt / self.tau_z) * (self.weight.mm(input_event_rate) + self.bias)
        self.event_rate = self.activation_function(self.soma_potential)

        return self.event_rate

    def feedback_update(self, target=None, dt=0.1):

        if target is not None:
            # delta_burst_prob = self.p_baseline * torch.tanh((target - self.event_rate_ma) / (self.event_rate_ma + self.small_vector))
            delta_burst_prob = self.p_baseline * (target - self.event_rate) * (1 - self.event_rate)
            self.burst_prob = self.p_baseline + delta_burst_prob
        else:
            self.burst_prob = self.p_baseline.detach().clone()

        self.burst_rate = self.burst_prob * self.event_rate

        self.event_rate_ma = self.event_rate_ma + (dt / self.tau_moving_avg) * (self.event_rate_cache - self.event_rate_ma)
        self.burst_rate_ma = self.burst_rate_ma + (dt / self.tau_moving_avg) * (self.burst_rate_cache - self.burst_rate_ma)
        # self.burst_prob_ma = self.burst_rate_ma / self.event_rate_ma

        return self.event_rate_cache, self.burst_rate_cache

    def weight_update(self, input_event_rate, dt=0.1):

        # H = ((torch.ones(self.out_features, 1) - self.e_max / (self.event_rate + 1e-8 * torch.ones(self.out_features, 1))) * (
        #             self.event_rate > self.e_max * torch.ones(self.out_features, 1))).mm(input_event_rate.t())

        self.bias = self.bias + (dt / self.tau_W) * ((self.burst_prob_cache - self.p_baseline) * self.event_rate_cache)

        # self.weight = self.weight + (dt / self.tau_W) * (((self.burst_prob_cache - self.burst_prob_ma_cache) * self.event_rate_cache) * input_event_rate.t() - self.lambda_ * H)
        self.weight = self.weight + (dt / self.tau_W) * (((self.burst_prob_cache - self.p_baseline) * self.event_rate_cache) * input_event_rate.t())

class ContinuousBurstCCNHiddenLayer:
    def __init__(self, in_features, out_features, next_features, p_baseline, device):
        self.in_features = in_features
        self.out_features = out_features
        self.p_baseline = p_baseline
        self.device = device

        self.soma_potential = torch.zeros((out_features, 1), device=device)
        self.soma_potential_cache = torch.zeros((out_features, 1), device=device)

        self.event_rate = torch.zeros((out_features, 1), device=device)
        self.event_rate_cache = torch.zeros((out_features, 1), device=device)

        self.event_rate_ma = torch.zeros((out_features, 1), device=device)
        self.event_rate_ma_cache = torch.zeros((out_features, 1), device=device)

        self.burst_rate = torch.zeros((out_features, 1), device=device)
        self.burst_rate_cache = torch.zeros((out_features, 1), device=device)

        self.burst_rate_ma = torch.zeros((out_features, 1), device=device)
        self.burst_rate_ma_cache = torch.zeros((out_features, 1), device=device)

        self.dendritic_potential = torch.zeros((out_features, 1), device=device)
        self.dendritic_potential_cache = torch.zeros((out_features, 1), device=device)

        self.burst_prob = torch.zeros((out_features, 1), device=device)
        self.burst_prob_cache = torch.zeros((out_features, 1), device=device)

        # self.burst_prob_ma = torch.zeros((out_features, 1), device=device)
        # self.burst_prob_ma_cache = torch.zeros((out_features, 1), device=device)

        self.beta = 4.0
        self.lambda_ = 0.05
        self.tau_u = 0.1
        self.tau_z = 0.1
        self.tau_e = self.tau_z
        self.tau_W = 100.0
        self.tau_moving_avg = 5.0
        self.e_max = 5.0

        self.weight = math.sqrt(2.0 / (in_features + out_features)) * torch.randn(out_features, in_features, device=device)
        self.bias = torch.zeros((out_features, 1), device=device)

        # if weight_transport:
        #     B = W2.t()
        # else:
        #     B = torch.randn(n_hidden, n_classes)
        #     B *= 0.1 * math.sqrt(2.0 / (n_hidden + n_classes))

        self.weight_Y = math.sqrt(2.0 / (out_features + next_features)) * torch.randn(next_features, out_features, device=device)
        self.weight_Q = -self.p_baseline * math.sqrt(2.0 / (out_features + next_features)) * torch.randn(next_features, out_features, device=device)

        self.activation_function = torch.sigmoid

        def h_for_sigmoid(x):
            return 1.0 - x

        self.h = h_for_sigmoid

    def cache_state(self):
        self.event_rate_cache = self.event_rate.detach().clone()
        self.burst_rate_cache = self.burst_rate.detach().clone()
        self.burst_prob_cache = self.burst_prob.detach().clone()

        self.event_rate_ma_cache = self.event_rate_ma.detach().clone()
        self.burst_rate_ma_cache = self.burst_rate_ma.detach().clone()
        # self.burst_prob_ma_cache = self.burst_prob_ma.detach().clone()

    def feedforward_update(self, input_event_rate, dt=0.1):
        self.soma_potential = (1. - dt / self.tau_z) * self.soma_potential + (dt / self.tau_z) * (self.weight.mm(input_event_rate) + self.bias)
        self.event_rate = self.activation_function(self.soma_potential)

        return self.event_rate

    def feedback_update(self, next_layer_event_rate_cache, next_layer_burst_rate_cache, dt=0.1):
        self.dendritic_potential = (1. - dt / self.tau_u) * self.dendritic_potential + (dt / self.tau_u) * (self.beta * self.h(self.event_rate_cache) * (self.weight_Q.T.mm(next_layer_event_rate_cache) + self.weight_Y.T.mm(next_layer_burst_rate_cache)))
        self.burst_prob = torch.sigmoid(self.dendritic_potential)
        self.burst_rate = self.burst_prob * self.event_rate

        self.event_rate_ma = self.event_rate_ma + (dt / self.tau_moving_avg) * (self.event_rate_cache - self.event_rate_ma)
        self.burst_rate_ma = self.burst_rate_ma + (dt / self.tau_moving_avg) * (self.burst_rate_cache - self.burst_rate_ma)
        # self.burst_prob_ma = self.burst_rate_ma / self.event_rate_ma

        return self.event_rate_cache, self.burst_rate_cache

    def weight_update(self, input_event_rate, dt=0.1):

        # H = ((torch.ones(self.out_features, 1) - self.e_max / (self.event_rate + 1e-8 * torch.ones(self.out_features, 1))) * (
        #             self.event_rate > self.e_max * torch.ones(self.out_features, 1))).mm(input_event_rate.t())

        self.bias = self.bias + (dt / self.tau_W) * ((self.burst_prob_cache - self.p_baseline) * self.event_rate_cache)
        # self.weight = self.weight + (dt / self.tau_W) * (((self.burst_prob_cache - self.burst_prob_ma_cache) * self.event_rate_cache) * input_event_rate.t() - self.lambda_ * H)
        self.weight = self.weight + (dt / self.tau_W) * (((self.burst_prob_cache - self.p_baseline) * self.event_rate_cache) * input_event_rate.t())
