import math
import torch


class SGDOptimiser:
    def __init__(self):
        pass

    def update_buffers(self, weight_grads, bias_grads):
        pass

    def compute_updates(self, lrs, weight_grads, bias_grads):
        weight_updates = []
        bias_updates = []

        for lr, weight_grad, bias_grad in zip(lrs, weight_grads, bias_grads):
            weight_update = -lr * weight_grad
            bias_update = -lr * bias_grad

            weight_updates.append(weight_update)
            bias_updates.append(bias_update)

        return weight_updates, bias_updates


class SGDMomentumOptimiser:
    def __init__(self, weight_sizes, bias_sizes, momentum=0.0, device='cpu'):
        self.momentum = momentum

        self.weight_m_buffers = [torch.zeros(weight_size, device=device) for weight_size in weight_sizes]
        self.bias_m_buffers = [torch.zeros(bias_size, device=device) for bias_size in bias_sizes]

    def update_buffers(self, weight_grads, bias_grads):
        for weight_m_buffer, weight_grad in zip(self.weight_m_buffers, weight_grads):
            # weight_m_buffer.mul_(self.momentum).add_(weight_grad, alpha=1 - self.momentum)
            weight_m_buffer.mul_(self.momentum).add_(weight_grad, alpha=1.0)

            # self.delta_weight = -lr * self.grad_weight / batch_size + momentum * self.delta_weight
            # weight_m_buffer = -0.3 * weight_grad + self.momentum * weight_m_buffer

        for bias_m_buffer, bias_grad in zip(self.bias_m_buffers, bias_grads):
            # bias_m_buffer.mul_(self.momentum).add_(bias_grad, alpha=1 - self.momentum)
            bias_m_buffer.mul_(self.momentum).add_(bias_grad, alpha=1.0)

            # bias_m_buffer = -0.3 * bias_grad + self.momentum * bias_m_buffer

    def compute_updates(self, lrs, weight_grads, bias_grads):

        weight_updates = []
        bias_updates = []

        for lr, weight_m_buffer, bias_m_buffer in zip(lrs, self.weight_m_buffers, self.bias_m_buffers):
            weight_update = -lr * weight_m_buffer
            bias_update = -lr * bias_m_buffer

            weight_updates.append(weight_update)
            bias_updates.append(bias_update)

        return weight_updates, bias_updates


class AdamOptimiser:
    def __init__(self, weight_sizes, bias_sizes, beta1=0.9, beta2=0.999, eps=1e-8, device='cpu'):
        # Init buffers
        self.weight_m_buffers = [torch.zeros(weight_size, device=device) for weight_size in weight_sizes]
        self.weight_v_buffers = [torch.zeros(weight_size, device=device) for weight_size in weight_sizes]

        self.bias_m_buffers = [torch.zeros(bias_size, device=device) for bias_size in bias_sizes]
        self.bias_v_buffers = [torch.zeros(bias_size, device=device) for bias_size in bias_sizes]

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.step_counter = 0

    def update_buffers(self, weight_grads, bias_grads):
        for weight_m_buffer, weight_v_buffer, weight_grad in zip(self.weight_m_buffers, self.weight_v_buffers, weight_grads):
            weight_m_buffer.mul_(self.beta1).add_(weight_grad, alpha=1 - self.beta1)
            weight_v_buffer.mul_(self.beta2).addcmul_(weight_grad, weight_grad, value=1 - self.beta2)

        for bias_m_buffer, bias_v_buffer, bias_grad in zip(self.bias_m_buffers, self.bias_v_buffers, bias_grads):
            bias_m_buffer.mul_(self.beta1).add_(bias_grad, alpha=1 - self.beta1)
            bias_v_buffer.mul_(self.beta2).addcmul_(bias_grad, bias_grad, value=1 - self.beta2)

    def compute_updates(self, lrs, weight_grads, bias_grads):
        self.step_counter += 1

        bias_correction1 = 1 - self.beta1 ** self.step_counter
        bias_correction2 = 1 - self.beta2 ** self.step_counter

        weight_updates = []
        for lr, weight_m_buffer, weight_v_buffer, weight_grad in zip(lrs, self.weight_m_buffers, self.weight_v_buffers, weight_grads):
            denom = (weight_v_buffer.sqrt() / math.sqrt(bias_correction2)).add_(self.eps)

            weight_update = -(lr/bias_correction1) * weight_m_buffer / denom
            weight_updates.append(weight_update)

        bias_updates = []
        for lr, bias_m_buffer, bias_v_buffer, bias_grad in zip(lrs, self.bias_m_buffers, self.bias_v_buffers, bias_grads):
            denom = (bias_v_buffer.sqrt() / math.sqrt(bias_correction2)).add_(self.eps)

            bias_update = -(lr / bias_correction1) * bias_m_buffer / denom
            bias_updates.append(bias_update)


        return weight_updates, bias_updates


class NeuronLeakOptimiser:
    def __init__(self, weight_sizes, bias_sizes, decay_constant=0.9, eps=1e-8, device='cpu'):
        # Init buffers
        self.G_buffers = [torch.zeros((weight_size[0], 1), device=device) for weight_size in weight_sizes]

        self.decay_constant = decay_constant
        self.eps = eps

    def update_buffers(self, weight_grads, bias_grads):
        for G_buffer, weight_grad, bias_grad in zip(self.G_buffers, weight_grads, bias_grads):
            param_grads = torch.cat([weight_grad, bias_grad.reshape(-1, 1)], dim=1)
            collapsed_param_grads = torch.mean(param_grads ** 2, dim=1, keepdim=True)

            G_buffer.mul_(self.decay_constant).add_(collapsed_param_grads, alpha=1 - self.decay_constant)

    def compute_updates(self, lrs, weight_grads, bias_grads):
        weight_updates = []
        bias_updates = []

        for lr, G_buffer, weight_grad, bias_grad in zip(lrs, self.G_buffers, weight_grads, bias_grads):
            denom = G_buffer.sqrt().add_(self.eps)

            weight_update = -lr * weight_grad / denom
            bias_update = -lr * bias_grad / denom.reshape(-1)

            weight_updates.append(weight_update)
            bias_updates.append(bias_update)

        return weight_updates, bias_updates


class LayerLeakOptimiser:
    def __init__(self, weight_sizes, bias_sizes, decay_constant=0.9, eps=1e-8, device='cpu'):
        # Init buffers
        self.G_buffers = [torch.zeros((1, 1), device=device) for _ in weight_sizes]

        self.decay_constant = decay_constant
        self.eps = eps

    def update_buffers(self, weight_grads, bias_grads):
        for G_buffer, weight_grad, bias_grad in zip(self.G_buffers, weight_grads, bias_grads):
            param_grads = torch.cat([weight_grad, bias_grad.reshape(-1, 1)], dim=1)
            collapsed_param_grads = torch.mean(param_grads ** 2).reshape(1,1)

            G_buffer.mul_(self.decay_constant).add_(collapsed_param_grads, alpha=1 - self.decay_constant)

    def compute_updates(self, lrs, weight_grads, bias_grads):
        weight_updates = []
        bias_updates = []

        for lr, G_buffer, weight_grad, bias_grad in zip(lrs, self.G_buffers, weight_grads, bias_grads):
            denom = G_buffer.sqrt().add_(self.eps)

            weight_update = -lr * weight_grad / denom
            bias_update = -lr * bias_grad / denom.reshape(-1)

            weight_updates.append(weight_update)
            bias_updates.append(bias_update)

        return weight_updates, bias_updates


class NetworkLeakOptimiser:
    def __init__(self, weight_sizes, bias_sizes, decay_constant=0.9, eps=1e-8, device='cpu'):
        # Init buffers
        self.G_buffer = torch.zeros((1, 1), device=device)

        self.decay_constant = decay_constant
        self.eps = eps

    def update_buffers(self, weight_grads, bias_grads):
        param_grads = torch.cat([weight_grad.flatten() for weight_grad in weight_grads] + [bias_grad.flatten() for bias_grad in bias_grads])
        collapsed_param_grads = torch.mean(param_grads ** 2).reshape((1, 1))

        self.G_buffer.mul_(self.decay_constant).add_(collapsed_param_grads, alpha=1 - self.decay_constant)

    def compute_updates(self, lrs, weight_grads, bias_grads):
        weight_updates = []
        bias_updates = []

        denom = self.G_buffer.sqrt().add_(self.eps)

        for lr, weight_grad, bias_grad in zip(lrs, weight_grads, bias_grads):
            weight_update = -lr * weight_grad / denom
            bias_update = -lr * bias_grad / denom.reshape(-1)

            weight_updates.append(weight_update)
            bias_updates.append(bias_update)

        return weight_updates, bias_updates


class NeuronLeakInverseOptimiser:
    def __init__(self, weight_sizes, bias_sizes, decay_constant=0.9, device='cpu'):
        # Init buffers
        self.G_buffers = [torch.zeros((weight_size[0], 1), device=device) for weight_size in weight_sizes]

        self.decay_constant = decay_constant

    def update_buffers(self, weight_grads, bias_grads):
        for G_buffer, weight_grad, bias_grad in zip(self.G_buffers, weight_grads, bias_grads):
            param_grads = torch.cat([weight_grad, bias_grad.reshape(-1, 1)], dim=1)
            collapsed_param_grads = torch.mean(param_grads ** 2, dim=1, keepdim=True)

            G_buffer.mul_(self.decay_constant).add_(collapsed_param_grads, alpha=1 - self.decay_constant)

    def compute_updates(self, lrs, weight_grads, bias_grads):
        weight_updates = []
        bias_updates = []

        for lr, G_buffer, weight_grad, bias_grad in zip(lrs, self.G_buffers, weight_grads, bias_grads):
            mod_factor = G_buffer.sqrt()

            weight_update = -lr * weight_grad * mod_factor
            bias_update = -lr * bias_grad * mod_factor.reshape(-1)

            weight_updates.append(weight_update)
            bias_updates.append(bias_update)

        return weight_updates, bias_updates


class LayerLeakInverseOptimiser:
    def __init__(self, weight_sizes, bias_sizes, decay_constant=0.9, device='cpu'):
        # Init buffers
        self.G_buffers = [torch.zeros((1, 1), device=device) for _ in weight_sizes]

        self.decay_constant = decay_constant

    def update_buffers(self, weight_grads, bias_grads):
        for G_buffer, weight_grad, bias_grad in zip(self.G_buffers, weight_grads, bias_grads):
            param_grads = torch.cat([weight_grad, bias_grad.reshape(-1, 1)], dim=1)
            collapsed_param_grads = torch.mean(param_grads ** 2).reshape(1,1)

            G_buffer.mul_(self.decay_constant).add_(collapsed_param_grads, alpha=1 - self.decay_constant)

    def compute_updates(self, lrs, weight_grads, bias_grads):
        weight_updates = []
        bias_updates = []

        for lr, G_buffer, weight_grad, bias_grad in zip(lrs, self.G_buffers, weight_grads, bias_grads):
            mod_factor = G_buffer.sqrt()

            weight_update = -lr * weight_grad * mod_factor
            bias_update = -lr * bias_grad * mod_factor.reshape(-1)

            weight_updates.append(weight_update)
            bias_updates.append(bias_update)

        return weight_updates, bias_updates


class NetworkLeakInverseOptimiser:
    def __init__(self, weight_sizes, bias_sizes, decay_constant=0.9, device='cpu'):
        # Init buffers
        self.G_buffer = torch.zeros((1, 1), device=device)

        self.decay_constant = decay_constant

    def update_buffers(self, weight_grads, bias_grads):
        param_grads = torch.cat([weight_grad.flatten() for weight_grad in weight_grads] + [bias_grad.flatten() for bias_grad in bias_grads])
        collapsed_param_grads = torch.mean(param_grads ** 2).reshape((1, 1))

        self.G_buffer.mul_(self.decay_constant).add_(collapsed_param_grads, alpha=1 - self.decay_constant)

    def compute_updates(self, lrs, weight_grads, bias_grads):
        weight_updates = []
        bias_updates = []

        mod_factor = self.G_buffer.sqrt()

        for lr, weight_grad, bias_grad in zip(lrs, weight_grads, bias_grads):
            weight_update = -lr * weight_grad * mod_factor
            bias_update = -lr * bias_grad * mod_factor.reshape(-1)

            weight_updates.append(weight_update)
            bias_updates.append(bias_update)

        return weight_updates, bias_updates


class SynapseIntegratorOptimiser:
    def __init__(self, weight_sizes, bias_sizes, eps=1e-8, device='cpu'):
        # Init buffers
        self.weight_G_buffers = [torch.zeros(weight_size, device=device) for weight_size in weight_sizes]
        self.bias_G_buffers = [torch.zeros(bias_size, device=device) for bias_size in bias_sizes]

        self.eps = eps

    def update_buffers(self, weight_grads, bias_grads):
        for weight_G_buffer, bias_G_buffer, weight_grad, bias_grad in zip(self.weight_G_buffers, self.bias_G_buffers, weight_grads, bias_grads):
            weight_G_buffer.add_(weight_grad ** 2)
            bias_G_buffer.add_(bias_grad ** 2)

    def compute_updates(self, lrs, weight_grads, bias_grads):
        weight_updates = []
        bias_updates = []

        for lr, weight_G_buffer, bias_G_buffer, weight_grad, bias_grad in zip(lrs, self.weight_G_buffers, self.bias_G_buffers, weight_grads, bias_grads):
            weight_denom = weight_G_buffer.sqrt().add_(self.eps)
            bias_denom = bias_G_buffer.sqrt().add_(self.eps)

            weight_update = -lr * weight_grad / weight_denom
            bias_update = -lr * bias_grad / bias_denom.reshape(-1)

            weight_updates.append(weight_update)
            bias_updates.append(bias_update)

        return weight_updates, bias_updates


class NeuronIntegratorOptimiser:
    def __init__(self, weight_sizes, bias_sizes, eps=1e-8, device='cpu'):
        # Init buffers
        self.G_buffers = [torch.zeros((weight_size[0], 1), device=device) for weight_size in weight_sizes]

        self.eps = eps

    def update_buffers(self, weight_grads, bias_grads):
        for G_buffer, weight_grad, bias_grad in zip(self.G_buffers, weight_grads, bias_grads):
            param_grads = torch.cat([weight_grad, bias_grad.reshape(-1, 1)], dim=1)
            collapsed_param_grads = torch.mean(param_grads ** 2, dim=1, keepdim=True)

            G_buffer.add_(collapsed_param_grads)

    def compute_updates(self, lrs, weight_grads, bias_grads):
        weight_updates = []
        bias_updates = []

        for lr, G_buffer, weight_grad, bias_grad in zip(lrs, self.G_buffers, weight_grads, bias_grads):
            denom = G_buffer.sqrt().add_(self.eps)

            weight_update = -lr * weight_grad / denom
            bias_update = -lr * bias_grad / denom.reshape(-1)

            weight_updates.append(weight_update)
            bias_updates.append(bias_update)

        return weight_updates, bias_updates


class LayerIntegratorOptimiser:
    def __init__(self, weight_sizes, bias_sizes, eps=1e-8, device='cpu'):
        # Init buffers
        self.G_buffers = [torch.zeros((1, 1), device=device) for _ in weight_sizes]

        self.eps = eps

    def update_buffers(self, weight_grads, bias_grads):
        for G_buffer, weight_grad, bias_grad in zip(self.G_buffers, weight_grads, bias_grads):
            param_grads = torch.cat([weight_grad, bias_grad.reshape(-1, 1)], dim=1)
            collapsed_param_grads = torch.mean(param_grads ** 2).reshape(1,1)

            G_buffer.add_(collapsed_param_grads)

    def compute_updates(self, lrs, weight_grads, bias_grads):
        weight_updates = []
        bias_updates = []

        for lr, G_buffer, weight_grad, bias_grad in zip(lrs, self.G_buffers, weight_grads, bias_grads):
            denom = G_buffer.sqrt().add_(self.eps)

            weight_update = -lr * weight_grad / denom
            bias_update = -lr * bias_grad / denom.reshape(-1)

            weight_updates.append(weight_update)
            bias_updates.append(bias_update)

        return weight_updates, bias_updates


class NetworkIntegratorOptimiser:
    def __init__(self, weight_sizes, bias_sizes, eps=1e-8, device='cpu'):
        # Init buffers
        self.G_buffer = torch.zeros((1, 1), device=device)

        self.eps = eps

    def update_buffers(self, weight_grads, bias_grads):
        param_grads = torch.cat([weight_grad.flatten() for weight_grad in weight_grads] + [bias_grad.flatten() for bias_grad in bias_grads])
        collapsed_param_grads = torch.mean(param_grads ** 2).reshape((1, 1))

        self.G_buffer.add_(collapsed_param_grads)

    def compute_updates(self, lrs, weight_grads, bias_grads):
        weight_updates = []
        bias_updates = []

        denom = self.G_buffer.sqrt().add_(self.eps)

        for lr, weight_grad, bias_grad in zip(lrs, weight_grads, bias_grads):
            weight_update = -lr * weight_grad / denom
            bias_update = -lr * bias_grad / denom.reshape(-1)

            weight_updates.append(weight_update)
            bias_updates.append(bias_update)

        return weight_updates, bias_updates


class NetworkCostOptimiser:
    def __init__(self, device='cpu'):
        # Init buffers
        self.cost_buffer = torch.zeros((1, 1), device=device)
        self.device = device

    def update_buffers(self, global_cost):
        self.cost_buffer.data = torch.tensor(global_cost, device=self.device).reshape((1, 1))

    def compute_updates(self, lrs, weight_grads, bias_grads):
        weight_updates = []
        bias_updates = []

        for lr, weight_grad, bias_grad in zip(lrs, weight_grads, bias_grads):
            weight_update = -lr * weight_grad * self.cost_buffer
            bias_update = -lr * bias_grad * self.cost_buffer.reshape(-1)

            weight_updates.append(weight_update)
            bias_updates.append(bias_update)

        return weight_updates, bias_updates