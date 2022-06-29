import os

import argparse

import random
import math

import numpy as np

import torch
import torch.nn.functional as F

import wandb

from datasets import get_mnist_dataset, get_xor_dataset, get_continuous_dataset
from modules.continuous_burstprop_networks import ContinuousBurstPropNetwork
from modules.continuous_burstccn_networks import ContinuousBurstCCNNetwork
from modules.networks import ANN

from helpers import similarity

def run_two_phase():
    prediction_time = 5.0
    teaching_time = 0.1 * prediction_time

    for epoch in range(n_epochs):
        print(f"Epoch #{epoch + 1}... ")
        loss_epoch = 0.0
        correct = 0

        for example_id, (inputs, target_class) in enumerate(train_data_loader):
            inputs, target_class = inputs.to(device), target_class.to(device)
            inputs = inputs.reshape(-1, 1)

            target = 0.01 + 0.98 * F.one_hot(target_class, num_classes=n_outputs).float().reshape(-1, 1)

            # for XOR
            # target = 0.01 + 0.98 * target_class.float().reshape(-1, 1)

            for t in range(int(prediction_time / dt)):
                net.prediction_update(inputs)

            prediction_before = net.layers[-1].event_rate.detach().clone()

            # Copy weights into ANN and backprop
            network_weights_before = [layer.weight.detach().clone() for layer in net.layers]
            network_biases_before = [layer.bias.detach().clone().squeeze() for layer in net.layers]
            ann.set_weights(network_weights_before, network_biases_before)
            ann.zero_grad()
            ann_output = ann.forward(inputs)
            ann_loss = mse_loss(ann_output, target.squeeze())
            ann_loss.backward()

            loss_before = loss(prediction_before, target)

            # print(f'*** Example {example_id}: {target_class.item()}')
            # print(f'Before: {torch.argmax(prediction_before)}, {loss_before}')

            if torch.argmax(prediction_before) == target_class.item():
                correct += 1

            for t in range(int(prediction_time / dt), int((prediction_time + teaching_time) / dt)):
                net.teaching_update(inputs, target)

            prediction_after = net.layers[-1].event_rate.detach().clone()
            delta_pred = (prediction_after - prediction_before).t()

            loss_after = loss(prediction_after, target)
            # print(f'After: {torch.argmax(prediction_after)}, {loss_after}')

            network_weights_after = [layer.weight.detach().clone() for layer in net.layers]
            network_biases_after = [layer.bias.detach().clone().squeeze() for layer in net.layers]

            # for t in range(int(prediction_time / dt)):
            #     net.prediction_update(inputs)
            #
            # prediction_after_training = net.layers[-1].event_rate.detach().clone()
            # print(f'After Training: {torch.argmax(prediction_after)}, {loss(prediction_after, target)}')

            loss_epoch += loss_before.cpu()
            example_costs.append(loss_before.cpu())

            weight_angles = [(180.0 / math.pi) * (torch.acos(similarity(-ann.linear_layers[i].weight.grad.flatten(), (
                        network_weights_after[i] - network_weights_before[i]).flatten()))) for i in
                             range(len(ann.linear_layers))]

            log_dict = {'example_id': example_id,
                        'example': example_id + len(train_data_loader) * epoch,
                        'loss_before': loss_before,
                        'loss_after': loss_after}

            log_dict.update({f'weight_angle ({i})': weight_angles[i] for i in range(len(weight_angles))})

            wandb.log(log_dict)

        train_error = 1.0 - correct / len(train_data_loader)
        loss_epoch_avg = loss_epoch / len(train_data_loader)
        wandb.log({'loss_epoch': loss_epoch_avg,
                   'train_error': train_error})


def run_one_phase(task_type, train_data_loader, num_training_examples):

    for example_id, (inputs, target) in enumerate(train_data_loader):
        inputs, target_class = inputs.to(device), target.to(device)
        inputs = inputs.reshape(-1, 1)

        if task_type == 'multi_classification':
            target = 0.01 + 0.98 * F.one_hot(target, num_classes=n_outputs).float().reshape(-1, 1)
        elif task_type == 'single_classification':
            # for XOR
            target = 0.01 + 0.98 * target.float().reshape(-1, 1)
        else:
            assert task_type == 'regression'

        # Copy weights into ANN and backprop
        network_weights_before = [layer.weight.detach().clone() for layer in net.layers]
        network_biases_before = [layer.bias.detach().clone().squeeze() for layer in net.layers]
        ann.set_weights(network_weights_before, network_biases_before)
        ann.zero_grad()
        ann_output = ann.forward(inputs)
        ann_loss = mse_loss(ann_output.reshape(-1, 1), target.reshape(-1, 1))
        ann_loss.backward()

        if example_id < 10000 or example_id >= num_training_examples + 10000:
            net.prediction_update(inputs)
        else:
            net.teaching_update(inputs, target)

        prediction = net.layers[-1].event_rate.detach().clone()

        loss_val = loss(prediction.reshape(-1, 1), target.reshape(-1, 1))

        network_weights_after = [layer.weight.detach().clone() for layer in net.layers]
        network_biases_after = [layer.bias.detach().clone().squeeze() for layer in net.layers]

        weight_angles = [(180.0 / math.pi) * (torch.acos(similarity(-ann.linear_layers[i].weight.grad.flatten(), (
                network_weights_after[i] - network_weights_before[i]).flatten()))) for i in
                         range(len(ann.linear_layers))]

        bias_angles = [(180.0 / math.pi) * (torch.acos(similarity(-ann.linear_layers[i].bias.grad.flatten(), (
                network_biases_after[i] - network_biases_before[i]).flatten()))) for i in
                       range(len(ann.linear_layers))]

        log_dict = {'example_id': example_id,
                    'loss': loss_val}

        if task_type == 'regression':
            log_dict.update({'output_0': prediction[0].item(),
                             'target_0': target[0].item()})

        log_dict.update({f'weight_angle ({i})': weight_angles[i] for i in range(len(weight_angles))})
        log_dict.update({f'bias_angle ({i})': bias_angles[i] for i in range(len(bias_angles))})

        wandb.log(log_dict)

        if example_id > num_training_examples + 20000:
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, help='Name of the run', required=True)
    parser.add_argument('--model_type', type=str, help='Type of model to use', required=True)

    parser.add_argument('--wandb_project', type=str, help='Wandb project name.', required=True)
    parser.add_argument('--wandb_entity', type=str, help='Wandb entity name.', required=True)

    parser.add_argument('--model_seed', type=int, help='The seed number', default=1)
    parser.add_argument('--num_training_examples', type=int, help='The number of examples to run', default=1000000)

    model_args = parser.parse_args()

    model_type = 'burstccn'
    wandb.init(project=run_args.wandb_project, entity=run_args.wandb_entity, name=model_args.run_name, config=model_args)

    n_inputs = 3
    n_hidden_layers = 1
    n_hidden_units = 50
    n_outputs = 1
    task_type = 'regression'

    # MNIST
    # n_inputs = 784
    # n_hidden_layers = 2
    # n_hidden_units = 200
    # n_outputs = 10

    # n_inputs = 2
    # n_hidden_layers = 1
    # n_hidden_units = 4
    # n_outputs = 1

    p_baseline = 0.5
    device = 'cpu'
    
    torch.manual_seed(model_args.model_seed)
    torch.cuda.manual_seed(model_args.model_seed)
    torch.cuda.manual_seed_all(model_args.model_seed)
    random.seed(model_args.model_seed)
    np.random.seed(model_args.model_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    if model_args.model_type == 'burstprop':
        net = ContinuousBurstPropNetwork(n_inputs, n_hidden_layers, n_hidden_units, n_outputs, p_baseline, device)
    elif model_args.model_type == 'burstccn':
        net = ContinuousBurstCCNNetwork(n_inputs, n_hidden_layers, n_hidden_units, n_outputs, p_baseline, device)

    ann = ANN(n_inputs, n_hidden_layers, n_hidden_units, n_outputs, device)

    # def loss(e, target, epsilon=1.e-7):
    #     output = torch.clip(e, epsilon, 1.0 - epsilon)
    #     s = torch.sum(-target * torch.log(output))
    #     return s

    def loss(e, target, epsilon=1.e-7):
        s = torch.sum((e - target) ** 2)
        return s

    #MNIST
    # data_dir = os.path.join(os.getcwd(), './Data')
    # train_data_loader, _, test_data_loader = get_mnist_dataset(data_dir, 1, 1, False, train_subset_size=1000)

    num_hidden_layers_of_task = n_hidden_layers

    task_seed = 1

    torch.manual_seed(task_seed)
    torch.cuda.manual_seed(task_seed)
    torch.cuda.manual_seed_all(task_seed)
    random.seed(task_seed)
    np.random.seed(task_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    train_data_loader = get_continuous_dataset(n_inputs, 1, False, num_hidden_layers_of_task, device)

    # train_data_loader, _, test_data_loader = get_xor_dataset()

    n_epochs = 1000
    dt = 0.1

    epoch_costs = []
    example_costs = []
    mse_loss = torch.nn.MSELoss()

    run_one_phase(task_type, train_data_loader, model_args.num_training_examples)
