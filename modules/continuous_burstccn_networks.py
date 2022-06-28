from .continuous_burstccn_layers import ContinuousBurstCCNHiddenLayer, ContinuousBurstCCNOutputLayer


class ContinuousBurstCCNNetwork:

    def __init__(self, n_inputs, n_hidden_layers, n_hidden_units, n_outputs, p_baseline, device):

        self.layers = []
        self.weight_transport = True

        if n_hidden_layers == 0:
            self.layers.append(ContinuousBurstCCNOutputLayer(n_inputs, n_outputs, p_baseline, device))
        elif n_hidden_layers == 1:
            self.layers.append(ContinuousBurstCCNHiddenLayer(n_inputs, n_hidden_units, n_outputs, p_baseline, device))
            self.layers.append(ContinuousBurstCCNOutputLayer(n_hidden_units, n_outputs, p_baseline, device))
        else:
            self.layers.append(ContinuousBurstCCNHiddenLayer(n_inputs, n_hidden_units, n_hidden_units, p_baseline, device))

            for i in range(1, n_hidden_layers - 1):
                self.layers.append(ContinuousBurstCCNHiddenLayer(n_hidden_units, n_hidden_units, n_hidden_units, p_baseline, device))

            self.layers.append(ContinuousBurstCCNHiddenLayer(n_hidden_units, n_hidden_units, n_outputs, p_baseline, device))
            self.layers.append(ContinuousBurstCCNOutputLayer(n_hidden_units, n_outputs, p_baseline, device))

        self.p_baseline = p_baseline

    def prediction_update(self, input_event_rate):
        event_rate = input_event_rate
        # for i, layer in enumerate(self.layers):
        for i in range(len(self.layers)):
            self.layers[i].cache_state()
            event_rate = self.layers[i].feedforward_update(event_rate)

        output_layer = self.layers[-1]
        # next_layer_burst_rate_cache = output_layer.burst_rate_cache
        next_layer_event_rate_cache, next_layer_burst_rate_cache = output_layer.feedback_update()
        # for i, layer in list(enumerate(self.layers))[-2::-1]:
        for i in range(len(self.layers) - 2, -1, -1):
            self.layers[i].cache_state()
            next_layer_event_rate_cache, next_layer_burst_rate_cache = self.layers[i].feedback_update(next_layer_event_rate_cache, next_layer_burst_rate_cache)

    def teaching_update(self, input_event_rate, target):
        event_rate = input_event_rate
        # for i, layer in enumerate(self.layers):
        for i in range(len(self.layers)):
            self.layers[i].cache_state()
            event_rate = self.layers[i].feedforward_update(event_rate)

        output_layer = self.layers[-1]
        next_layer_event_rate_cache, next_layer_burst_rate_cache = output_layer.feedback_update(target)

        # for i, layer in list(enumerate(self.layers))[-2::-1]:
        for i in range(len(self.layers) - 2, -1, -1):
            self.layers[i].cache_state()
            next_layer_event_rate_cache, next_layer_burst_rate_cache = self.layers[i].feedback_update(next_layer_event_rate_cache, next_layer_burst_rate_cache)

        first_hidden_layer = self.layers[0]
        first_hidden_layer.weight_update(input_event_rate)

        # for i, layer in list(enumerate(self.layers))[1:]:
        for i in range(1, len(self.layers)):
            self.layers[i].weight_update(self.layers[i-1].event_rate_cache)

        if self.weight_transport:
            for i in range(len(self.layers) - 1):
                self.layers[i].weight_Y = self.layers[i+1].weight.detach().clone()
                self.layers[i].weight_Q = -self.p_baseline * self.layers[i+1].weight.detach().clone()
