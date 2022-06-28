from abc import abstractmethod, ABC

import datetime
import os
import shutil

import numpy as np

import torch
import torch.nn.functional as F

from tqdm import tqdm
import wandb

from modules.networks import MNISTNetFA
from modules.networks_burstprop import BurstProp, ConvBurstProp
from modules.networks_burstccn import BurstCCN, ConvBurstCCN
from modules.networks_edn import MNISTNetEDN

from modules.optimisers import AdamOptimiser, SGDOptimiser, SGDMomentumOptimiser, NeuronLeakOptimiser, \
    LayerIntegratorOptimiser, \
    NetworkLeakOptimiser, SynapseIntegratorOptimiser, NeuronIntegratorOptimiser, NetworkIntegratorOptimiser, \
    NetworkCostOptimiser


class BioModelTrainer(ABC):
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Overwritten by parse_model_params
        self.model = None
        self.feedforward_noise = None
        self.no_teaching_signal = False

        self.log_frequency = 200

    @abstractmethod
    def parse_model_params(self, parser):
        pass

    @abstractmethod
    def set_config(self, model_args):
        pass

    @abstractmethod
    def update_model_weights(self, global_cost=None):
        pass

    def log_state(self):
        pass

    def log_layer_states(self):
        pass

    def train(self, train_loader):
        self.model.train()

        train_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader)
        for batch_index, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(self.model.device), targets.to(self.model.device)

            t = F.one_hot(targets, num_classes=10).float()

            # Noiseless with teacher backwards phase to log layer angles and layer updates
            if batch_index % self.log_frequency == 0 and self.config.log_mode != 'minimal':
                if self.feedforward_noise is not None:
                    self.model.set_forward_noise(None)

                outputs = self.model(inputs)
                self.model.backward(t)
                self.log_state()

            # Noiseless backwards phase with or without teacher to log layer states
            if batch_index % self.log_frequency == 0 and self.config.log_mode != 'minimal' and self.config.log_mode != 'no_states':
                if self.feedforward_noise is not None:
                    self.model.set_forward_noise(None)

                outputs = self.model(inputs)
                if self.no_teaching_signal:
                    self.model.backward(None)
                else:
                    self.model.backward(t)
                self.log_layer_states()

            # Log the batch_index and any other previously uncommitted states
            if batch_index % self.log_frequency == 0 and self.config.log_mode != 'minimal':
                wandb.log({'batch_index': batch_index})

            if self.feedforward_noise is not None:
                self.model.set_forward_noise(self.feedforward_noise)

            outputs = self.model(inputs)
            if self.no_teaching_signal:
                self.model.backward(None)
            else:
                self.model.backward(t)

            loss = self.model.loss(outputs, t)

            self.update_model_weights(global_cost=loss.item())

            train_loss += loss
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar.set_description(
                "Train Loss: {:.3f} | Acc: {:.3f}% ({:d}/{:d})".format(train_loss / (batch_index + 1),
                                                                       100 * correct / total, correct, total))

        return 100.0 * (1.0 - correct / total), train_loss / (batch_index + 1)

    def test(self, test_loader):
        self.model.eval()

        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            progress_bar = tqdm(test_loader)
            for batch_idx, (inputs, targets) in enumerate(progress_bar):
                inputs, targets = inputs.to(self.model.device), targets.to(self.model.device)

                t = F.one_hot(targets, num_classes=10).float()

                outputs = self.model(inputs)

                loss = self.model.loss(outputs, t)

                test_loss += loss.item()

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar.set_description(
                    "Test Loss: {:.3f} | Acc: {:.3f}% ({:d}/{:d})".format(test_loss / (batch_idx + 1),
                                                                          100 * correct / total, correct, total))

        return 100 * (1.0 - correct / total), test_loss / (batch_idx + 1)

    def get_save_strings(self):
        save_strings = []

        # save a human-readable text file containing simulation details
        timestamp = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
        save_strings.append(f"Simulation run @ {timestamp}")
        save_strings.append(f"Number of epochs: {self.config.n_epochs}")
        save_strings.append(f"Batch size: {self.config.batch_size}")
        save_strings.append(f"Using validation set: {self.config.use_validation}")

        save_strings.append(f"Feedforward learning rates: {self.config.lr}")
        save_strings.append(f"Feedback weight initialization scale: {self.config.Y_scale}")
        save_strings.append(f"Momentum: {self.config.momentum}")
        save_strings.append(f"Weight decay: {self.config.weight_decay}")
        save_strings.append(f"Feedback weight learning: {self.config.Y_learning}")

        return save_strings

    def save_training_setup(self):
        run_directory = os.path.join(self.config.working_directory, 'results', self.config.run_name)

        if not os.path.exists(run_directory):
            os.makedirs(run_directory)

        save_strings = self.get_save_strings()

        with open(os.path.join(run_directory, "params.txt"), "w") as f:
            for save_string in save_strings:
                f.write(save_string + "\n")

        shutil.copyfile(os.path.join(self.config.working_directory, 'helpers.py'),
                        os.path.join(run_directory, 'helpers.py'))
        shutil.copyfile(os.path.join(self.config.working_directory, 'datasets.py'),
                        os.path.join(run_directory, 'datasets.py'))
        shutil.copyfile(os.path.join(self.config.working_directory, "modules/networks.py"),
                        os.path.join(run_directory, "networks.py"))
        shutil.copyfile(os.path.join(self.config.working_directory, "modules/layers.py"),
                        os.path.join(run_directory, "layers.py"))
        torch.save(self.model, os.path.join(run_directory, "initial_model.pth"))


class BurstPropTrainer(BioModelTrainer):
    def __init__(self):
        super().__init__()

    def parse_model_params(self, parser):
        parser.add_argument("--p_baseline", type=float, help="Output layer baseline burst probability",
                            default=0.2)
        parser.add_argument("--n_hidden_layers", type=int,
                            help="Number of hidden layers",
                            default=3)
        parser.add_argument("--n_hidden_units", type=int,
                            help="Number of hidden units in each layer",
                            default=500)

        parser.add_argument('--recurrent_input', default=True, help="Whether to use recurrent input",
                            type=lambda x: (str(x).lower() == 'true'))

        parser.add_argument('--Y_learning', default=False, help="Whether to update feedback weights",
                            type=lambda x: (str(x).lower() == 'true'))
        parser.add_argument('--recurrent_learning', default=True, help="Whether to update recurrent weights",
                            type=lambda x: (str(x).lower() == 'true'))

        parser.add_argument("--lr", help="Learning rate for hidden layers", type=float, default=0.1)
        parser.add_argument("--Y_lr", type=float, default=0.0)
        parser.add_argument("--recurrent_lr", help="Learning rate for recurrent weights", type=float,
                            default=0.0001)

        parser.add_argument("--momentum", type=float, help="Momentum", default=0.0)
        parser.add_argument("--weight_decay", type=float, help="Weight decay", default=0.0)

        parser.add_argument("--Y_mode", type=str, help="Must be 'tied', 'symmetric_init' or 'random_init",
                            default='random_init')
        parser.add_argument("--Y_scale", type=float, help="Scale of the feedback weights.", default=1.0)
        parser.add_argument("--recurrent_scale", help="Scale of initial recurrent weights", type=float,
                            default=0.01)

        parser.add_argument("--no_teaching_signal", default=False, type=lambda x: (str(x).lower() == 'true'))
        parser.add_argument("--feedforward_noise", default=None, type=float)

        parser.add_argument("--use_conv", help="Use conv layers", default=False,
                            type=lambda x: (str(x).lower() == 'true'))

        model_args, _ = parser.parse_known_args()

        return model_args

    def set_config(self, config):
        self.config = config

        if config.use_conv:
            self.lr = [config.lr] * 5
        else:
            self.lr = [config.lr] * (config.n_hidden_layers + 1)

        self.momentum = config.momentum
        self.weight_decay = config.weight_decay
        self.recurrent_lr = config.recurrent_lr
        self.batch_size = config.batch_size
        self.no_teaching_signal = config.no_teaching_signal
        self.feedforward_noise = config.feedforward_noise

        if config.use_conv:
            self.model = ConvBurstProp(input_channels=3, p_baseline=config.p_baseline, Y_mode=config.Y_mode,
                                       Y_scale=config.Y_scale,
                                       Y_learning=config.Y_learning, recurrent_input=config.recurrent_input,
                                       recurrent_learning=config.recurrent_learning,
                                       recurrent_scale=config.recurrent_scale, device=self.device)
        else:
            self.model = BurstProp(p_baseline=config.p_baseline,
                                   n_hidden_layers=config.n_hidden_layers,
                                   n_hidden_units=config.n_hidden_units,
                                   recurrent_input=config.recurrent_input,
                                   Y_learning=config.Y_learning,
                                   Y_mode=config.Y_mode,
                                   Y_scale=config.Y_scale,
                                   recurrent_learning=config.recurrent_learning,
                                   recurrent_scale=config.recurrent_scale,
                                   device=self.device).to(self.device)

    def update_model_weights(self, global_cost=None):
        self.model.update_weights(lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay,
                                  recurrent_lr=self.recurrent_lr, batch_size=self.batch_size)

    def log_state(self):
        super().log_state()
        state = dict()

        if self.config.log_mode == 'all' or self.config.log_mode == 'no_states':
            layer_angles_dict = self.get_layer_angles()
            layer_update_magnitudes_dict = self.get_layer_update_magnitudes()
            loggable_update_magnitudes_dict = {k: wandb.Histogram(np_histogram=(v, np.array(range(len(v) + 1)))) for
                                               k, v in
                                               layer_update_magnitudes_dict.items()}

            state = state | layer_angles_dict | loggable_update_magnitudes_dict

        wandb.log(state, commit=False)

    def log_layer_states(self):
        state = dict()

        if self.config.log_mode == 'all':
            layer_states_dict = self.get_layer_states()
            loggable_layer_states_dict = {k: wandb.Histogram(v) for k, v in layer_states_dict.items()}

            state = state | loggable_layer_states_dict

        wandb.log(state, commit=False)

    def get_layer_angles(self):
        W_Y_angles = self.model.weight_angles_W_Y()
        W_Y_angles_dict = {f'angle_W_Y ({i})': W_Y_angles[i] for i in range(len(W_Y_angles))}

        bp_angles = self.model.bp_angles()
        bp_angles_dict = {f'angle_bp ({i})': bp_angles[i] for i in range(len(bp_angles))}
        fa_angles = self.model.fa_angles()
        fa_angles_dict = {f'angle_fa ({i})': fa_angles[i] for i in range(len(fa_angles))}
        fa_to_bp_angles = self.model.fa_to_bp_angles()
        fa_to_bp_angles_dict = {f'angle_fa_to_bp ({i})': fa_to_bp_angles[i] for i in range(len(fa_to_bp_angles))}

        global_bp_angle = self.model.global_bp_angle()
        global_fa_angle = self.model.global_fa_angle()
        global_fa_to_bp_angle = self.model.global_fa_to_bp_angle()
        global_angles_dict = {'global_bp_angle': global_bp_angle,
                              'global_fa_angle': global_fa_angle,
                              'global_fa_to_bp_angle': global_fa_to_bp_angle}

        layer_angles_dict = W_Y_angles_dict | bp_angles_dict | fa_angles_dict | fa_to_bp_angles_dict | global_angles_dict
        return layer_angles_dict

    def get_layer_update_magnitudes(self):
        bp_grad_magnitudes = self.model.bp_grad_magnitudes()
        grad_magnitudes = self.model.grad_magnitudes()

        layer_update_magnitudes_dict = {'bp_grad_magnitudes': bp_grad_magnitudes,
                                        'grad_magnitudes': grad_magnitudes}
        return layer_update_magnitudes_dict

    def get_layer_states(self):
        layer_states_dict = dict()
        # Get hidden layer states
        for i in range(len(self.model.classification_layers) - 1):
            layer = self.model.classification_layers[i]
            layer_states_dict[f"hidden{i + 1}.event_rate"] = layer.e.flatten().cpu().numpy()
            layer_states_dict[f"hidden{i + 1}.burst_prob"] = layer.p_t.flatten().cpu().numpy()
            layer_states_dict[f"hidden{i + 1}.burst_rate"] = layer.b_t.flatten().cpu().numpy()

        # Get output layer states
        output_layer = self.model.classification_layers[-1]
        layer_states_dict["output.event_rate"] = output_layer.e.flatten().cpu().numpy()
        layer_states_dict["output.burst_prob"] = output_layer.p_t.flatten().cpu().numpy()
        layer_states_dict["output.burst_rate"] = output_layer.b_t.flatten().cpu().numpy()

        return layer_states_dict

    def get_save_strings(self):
        save_strings = super().get_save_strings()

        save_strings.append(f"Output layer baseline burst probability: {self.config.p_baseline}")
        save_strings.append(f"Recurrent input: {self.config.recurrent_input}")
        save_strings.append(f"Recurrent weight initialization standard deviation: {self.config.recurrent_scale}")
        save_strings.append(f"Recurrent weight learning: {self.config.recurrent_learning}")
        save_strings.append(f"Recurrent weight learning rate: {self.config.recurrent_lr}")

        return save_strings

    def save_training_setup(self):
        super().save_training_setup()
        run_directory = os.path.join(self.config.working_directory, 'results', self.config.run_name)
        shutil.copyfile(os.path.join(self.config.working_directory, 'model_trainers.py'),
                        os.path.join(run_directory, 'model_trainers.py'))
        shutil.copyfile(os.path.join(self.config.working_directory, "modules/networks_burstprop.py"),
                        os.path.join(run_directory, "networks_burstprop.py"))
        shutil.copyfile(os.path.join(self.config.working_directory, "modules/layers_burstprop.py"),
                        os.path.join(run_directory, "layers_burstprop.py"))


class BurstCCNTrainer(BioModelTrainer):
    def __init__(self):
        super().__init__()

    def parse_model_params(self, parser):
        parser.add_argument("--p_baseline", type=float, help="Output layer baseline burst probability",
                            default=0.5)
        parser.add_argument("--n_hidden_layers", type=int,
                            help="Number of hidden layers",
                            default=3)
        parser.add_argument("--n_hidden_units", type=int,
                            help="Number of hidden units in each layer",
                            default=500)

        parser.add_argument('--Y_learning', default=False, help="Whether to update Y feedback weights",
                            type=lambda x: (str(x).lower() == 'true'))
        parser.add_argument('--Q_learning', default=False, help="Whether to update Y feedback weights",
                            type=lambda x: (str(x).lower() == 'true'))

        parser.add_argument("--lr", help="Learning rate for hidden layers", type=float, default=0.1)
        parser.add_argument("--Y_lr", type=float, default=0.001)
        parser.add_argument("--Q_lr", type=float, default=0.001)

        parser.add_argument("--momentum", type=float, help="Momentum", default=0.0)
        parser.add_argument("--weight_decay", type=float, help="Weight decay", default=0.0)

        parser.add_argument("--Y_mode", type=str, help="Must be 'tied', 'symmetric_init' or 'random_init",
                            default='random_init')
        parser.add_argument("--Y_scale", type=float, help="Scale of the feedback weights.", default=1.0)

        parser.add_argument("--Q_mode", type=str, help="Must be 'tied', 'symmetric_init' or 'random_init",
                            default='symmetric_init')
        parser.add_argument("--Q_scale", type=float, help="", default=1.0)

        parser.add_argument("--no_teaching_signal", default=False, type=lambda x: (str(x).lower() == 'true'))
        parser.add_argument("--feedforward_noise", default=None, type=float)

        parser.add_argument("--optimiser_type", type=str, help="Optimiser type", default='SGD')

        parser.add_argument("--use_conv", help="Use conv layers", default=False,
                            type=lambda x: (str(x).lower() == 'true'))

        model_args, _ = parser.parse_known_args()

        return model_args

    def set_config(self, config):
        self.config = config

        if config.use_conv:
            self.lr = [config.lr] * 5
            self.lr_Y = [config.Y_lr] * 4 + [None]
            self.lr_Q = [config.Q_lr] * 4 + [None]
        else:
            self.lr = [config.lr] * (config.n_hidden_layers + 1)
            self.lr_Y = [config.Y_lr] * config.n_hidden_layers + [None]
            self.lr_Q = [config.Q_lr] * config.n_hidden_layers + [None]

        self.momentum = config.momentum
        self.weight_decay = config.weight_decay

        self.batch_size = config.batch_size
        self.no_teaching_signal = config.no_teaching_signal
        self.feedforward_noise = config.feedforward_noise

        if config.use_conv:
            self.model = ConvBurstCCN(n_inputs=None, n_outputs=10, p_baseline=0.5, n_hidden_layers=None,
                                      n_hidden_units=None, Y_mode=config.Y_mode, Q_mode=config.Q_mode,
                                      Y_scale=config.Y_scale,
                                      Q_scale=config.Q_scale, Y_learning=config.Y_learning,
                                      Q_learning=config.Q_learning, device=self.device)
        else:
            self.model = BurstCCN(
                n_inputs=784,
                n_outputs=10,
                p_baseline=config.p_baseline,
                n_hidden_layers=config.n_hidden_layers,
                n_hidden_units=config.n_hidden_units,
                Y_learning=config.Y_learning,
                Y_mode=config.Y_mode,
                Y_scale=config.Y_scale,
                Q_learning=config.Q_learning,
                Q_mode=config.Q_mode,
                Q_scale=config.Q_scale,
                device=self.device).to(self.device)

        if config.optimiser_type == 'SGD':
            if self.momentum == 0.0:
                self.optimiser = SGDOptimiser()
            else:
                self.optimiser = SGDMomentumOptimiser(
                    weight_sizes=[self.model.classification_layers[i].weight.shape for i in
                                  range(len(self.model.classification_layers))],
                    bias_sizes=[self.model.classification_layers[i].bias.shape for i in
                                range(len(self.model.classification_layers))],
                    momentum=self.momentum,
                    device=self.device)
        elif config.optimiser_type == 'sAdagrad':
            self.optimiser = SynapseIntegratorOptimiser(
                weight_sizes=[self.model.classification_layers[i].weight.shape for i in
                              range(len(self.model.classification_layers))],
                bias_sizes=[self.model.classification_layers[i].bias.shape for i in
                            range(len(self.model.classification_layers))],
                device=self.device)
        elif config.optimiser_type == 'nAdagrad':
            self.optimiser = NeuronIntegratorOptimiser(
                weight_sizes=[self.model.classification_layers[i].weight.shape for i in
                              range(len(self.model.classification_layers))],
                bias_sizes=[self.model.classification_layers[i].bias.shape for i in
                            range(len(self.model.classification_layers))],
                device=self.device)
        elif config.optimiser_type == 'lAdagrad':
            self.optimiser = LayerIntegratorOptimiser(
                weight_sizes=[self.model.classification_layers[i].weight.shape for i in
                              range(len(self.model.classification_layers))],
                bias_sizes=[self.model.classification_layers[i].bias.shape for i in
                            range(len(self.model.classification_layers))],
                device=self.device)
        elif config.optimiser_type == 'fAdagrad':
            self.optimiser = NetworkIntegratorOptimiser(
                weight_sizes=[self.model.classification_layers[i].weight.shape for i in
                              range(len(self.model.classification_layers))],
                bias_sizes=[self.model.classification_layers[i].bias.shape for i in
                            range(len(self.model.classification_layers))],
                device=self.device)
        elif config.optimiser_type == 'Adam':
            self.optimiser = AdamOptimiser(
                weight_sizes=[self.model.classification_layers[i].weight.shape for i in
                              range(len(self.model.classification_layers))],
                bias_sizes=[self.model.classification_layers[i].bias.shape for i in
                            range(len(self.model.classification_layers))],
                device=self.device)
        elif config.optimiser_type == 'globalCost':
            self.optimiser = NetworkCostOptimiser(device=self.device)

    def update_model_weights(self, global_cost=None):
        self.model.update_weights(lrs=self.lr, lrs_Y=self.lr_Y, lrs_Q=self.lr_Q, optimiser=self.optimiser,
                                  global_cost=global_cost, weight_decay=self.weight_decay, use_backprop=False, use_feedback_alignment=False)

    def log_state(self):
        super().log_state()
        state = dict()

        if self.config.log_mode == 'all' or self.config.log_mode == 'no_states':
            layer_angles_dict = self.get_layer_angles()
            layer_update_magnitudes_dict = self.get_layer_update_magnitudes()
            loggable_update_magnitudes_dict = {k: wandb.Histogram(np_histogram=(v, np.array(range(len(v) + 1)))) for
                                               k, v in
                                               layer_update_magnitudes_dict.items()}

            state = state | layer_angles_dict | loggable_update_magnitudes_dict

        wandb.log(state, commit=False)

    def log_layer_states(self):
        state = dict()

        if self.config.log_mode == 'all':
            layer_states_dict = self.get_layer_states()

            state = state | layer_states_dict

        wandb.log(state, commit=False)

    def get_layer_angles(self):
        W_Y_angles = self.model.weight_angles_W_Y()
        W_Y_angles_dict = {f'angle_W_Y ({i})': W_Y_angles[i] for i in range(len(W_Y_angles))}
        Q_Y_angles = self.model.weight_angles_Q_Y()
        Q_Y_angles_dict = {f'angle_Q_Y ({i})': Q_Y_angles[i] for i in range(len(Q_Y_angles))}

        bp_angles = self.model.bp_angles()
        bp_angles_dict = {f'angle_bp ({i})': bp_angles[i] for i in range(len(bp_angles))}
        fa_angles = self.model.fa_angles()
        fa_angles_dict = {f'angle_fa ({i})': fa_angles[i] for i in range(len(fa_angles))}
        fa_to_bp_angles = self.model.fa_to_bp_angles()
        fa_to_bp_angles_dict = {f'angle_fa_to_bp ({i})': fa_to_bp_angles[i] for i in range(len(fa_to_bp_angles))}

        global_bp_angle = self.model.global_bp_angle()
        global_fa_angle = self.model.global_fa_angle()
        global_fa_to_bp_angle = self.model.global_fa_to_bp_angle()
        global_Q_Y_angle = self.model.global_weight_angle_Q_Y()

        global_angles_dict = {'global_bp_angle': global_bp_angle,
                              'global_fa_angle': global_fa_angle,
                              'global_fa_to_bp_angle': global_fa_to_bp_angle,
                              'global_Q_Y_angle': global_Q_Y_angle}

        layer_angles_dict = W_Y_angles_dict | Q_Y_angles_dict | bp_angles_dict | fa_angles_dict | fa_to_bp_angles_dict | global_angles_dict
        return layer_angles_dict

    def get_layer_update_magnitudes(self):
        bp_grad_magnitudes = self.model.bp_grad_magnitudes()
        grad_magnitudes = self.model.grad_magnitudes()

        layer_update_magnitudes_dict = {'bp_grad_magnitudes': bp_grad_magnitudes,
                                        'grad_magnitudes': grad_magnitudes}
        return layer_update_magnitudes_dict

    def get_layer_states(self):
        histogram_states_dict = dict()
        layer_metrics_dict = dict()
        # Get hidden layer states
        for i in range(len(self.model.classification_layers) - 1):
            layer = self.model.classification_layers[i]
            histogram_states_dict[f"hidden{i + 1}.event_rate"] = layer.e.flatten().cpu().numpy()
            histogram_states_dict[f"hidden{i + 1}.burst_prob"] = layer.p_t.flatten().cpu().numpy()
            histogram_states_dict[f"hidden{i + 1}.apical"] = layer.apic.flatten().cpu().numpy()
            histogram_states_dict[f"hidden{i + 1}.burst_rate"] = layer.b_t.flatten().cpu().numpy()

            layer_metrics_dict[f"hidden{i + 1}.apical_variance"] = np.var(layer.apic.flatten().cpu().numpy())
            layer_metrics_dict[f"hidden{i + 1}.apical_mean"] = np.mean(layer.apic.flatten().cpu().numpy())
            layer_metrics_dict[f"hidden{i + 1}.apical_magnitude"] = np.mean(np.abs(layer.apic.flatten().cpu().numpy()))

            layer_metrics_dict[f"hidden{i + 1}.burst_prob_variance"] = np.var(layer.p_t.flatten().cpu().numpy())
            layer_metrics_dict[f"hidden{i + 1}.burst_prob_mean"] = np.mean(layer.p_t.flatten().cpu().numpy())
            layer_metrics_dict[f"hidden{i + 1}.burst_prob_diff_magnitude"] = np.mean(
                np.abs(layer.p_t.flatten().cpu().numpy() - 0.5))

        # Get output layer states
        output_layer = self.model.classification_layers[-1]
        histogram_states_dict["output.event_rate"] = output_layer.e.flatten().cpu().numpy()
        histogram_states_dict["output.burst_prob"] = output_layer.p_t.flatten().cpu().numpy()
        histogram_states_dict["output.burst_rate"] = output_layer.b_t.flatten().cpu().numpy()

        histogram_states_dict = {k: wandb.Histogram(v) for k, v in histogram_states_dict.items()}

        return histogram_states_dict | layer_metrics_dict

    def get_save_strings(self):
        save_strings = super().get_save_strings()

        save_strings.append(f"Baseline burst probability: {self.config.p_baseline}")

        save_strings.append(f"Q weight learning: {self.config.Q_learning}")
        save_strings.append(f"Q weight learning rate: {self.config.Q_lr}")
        save_strings.append(f"Q weight initialization standard deviation: {self.config.Q_scale}")

        return save_strings

    def save_training_setup(self):
        super().save_training_setup()
        run_directory = os.path.join(self.config.working_directory, 'results', self.config.run_name)
        shutil.copyfile(os.path.join(self.config.working_directory, 'model_trainers.py'),
                        os.path.join(run_directory, 'model_trainers.py'))
        shutil.copyfile(os.path.join(self.config.working_directory, "modules/networks_burstccn.py"),
                        os.path.join(run_directory, "networks_burstccn.py"))
        shutil.copyfile(os.path.join(self.config.working_directory, "modules/layers_burstccn.py"),
                        os.path.join(run_directory, "layers_burstccn.py"))


class EDNTrainer(BioModelTrainer):
    def __init__(self):
        super().__init__()

    def parse_model_params(self, parser):
        parser.add_argument("--n_hidden_layers", type=int,
                            help="Number of hidden layers",
                            default=3)
        parser.add_argument("--n_hidden_units", type=int,
                            help="Number of hidden units in each layer",
                            default=500)

        parser.add_argument("--lr", help="Learning rate", type=float, default=0.1)
        parser.add_argument("--lr_Y", help="Learning rate for feedback weights", type=float, default=0.1)

        parser.add_argument("--Y_mode", type=str, help="Must be 'tied', 'symmetric_init' or 'random_init",
                            default='random_init')
        parser.add_argument("--Y_scale", type=float, help="Scale of the feedback weights.", default=1.0)
        parser.add_argument('--Y_learning', default=False, help="Whether to update Y feedback weights",
                            type=lambda x: (str(x).lower() == 'true'))

        parser.add_argument("--intn_lr_scale", type=float, help="", default=2.0)
        parser.add_argument("--intn_lr_override", type=float, help="", default=None)

        parser.add_argument("--intn_mode", type=str, help="Must be 'tied', 'symmetric_init' or 'random_init",
                            default='symmetric_init')
        parser.add_argument("--intn_scale", type=float, help="", default=1.0)

        parser.add_argument("--lambda_output", type=float, help="lambda output", default=0.1)
        parser.add_argument("--lambda_intn", type=float, help="lambda interneuron", default=0.1)
        parser.add_argument("--lambda_hidden", type=float, help="lambda hidden", default=0.1)

        parser.add_argument("--momentum", type=float, help="Momentum", default=0.0)
        parser.add_argument("--weight_decay", type=float, help="Weight decay", default=0.0)

        parser.add_argument("--optimiser_type", type=str, help="Optimiser type", default='SGD')

        model_args, _ = parser.parse_known_args()

        return model_args

    def set_config(self, config):
        self.config = config

        self.lr = [config.lr] * (config.n_hidden_layers + 1)
        self.lr_Y = [config.lr_Y] * (config.n_hidden_layers + 1)
        self.momentum = config.momentum
        self.weight_decay = config.weight_decay

        self.batch_size = config.batch_size

        if config.intn_lr_override is not None:
            self.lr_pyr_intn = [config.intn_lr_override] * config.n_hidden_layers + [None]
            self.lr_intn_pyr = [config.intn_lr_override] * config.n_hidden_layers + [None]
        else:
            self.lr_pyr_intn = [config.intn_lr_scale * config.lr] * config.n_hidden_layers + [None]
            self.lr_intn_pyr = [config.intn_lr_scale * config.lr] * config.n_hidden_layers + [None]

        self.model = MNISTNetEDN(n_hidden_layers=config.n_hidden_layers, n_hidden_units=config.n_hidden_units,
                                 lambda_output=config.lambda_output, lambda_intn=config.lambda_intn,
                                 lambda_hidden=config.lambda_hidden, Y_mode=config.Y_mode,
                                 Y_scale=config.Y_scale, Y_learning=config.Y_learning,
                                 intn_feedback_mode=config.intn_mode,
                                 intn_feedback_scale=config.intn_scale, device=self.device).to(self.device)

        if config.optimiser_type == 'SGD':
            if self.momentum == 0.0:
                self.optimiser = SGDOptimiser()
            else:
                self.optimiser = SGDMomentumOptimiser(
                    weight_sizes=[self.model.classification_layers[i].weight.shape for i in
                                  range(len(self.model.classification_layers))],
                    bias_sizes=[self.model.classification_layers[i].bias.shape for i in
                                range(len(self.model.classification_layers))],
                    momentum=self.momentum,
                    device=self.device)
        elif config.optimiser_type == 'sAdagrad':
            self.optimiser = SynapseIntegratorOptimiser(
                weight_sizes=[self.model.classification_layers[i].weight.shape for i in
                              range(len(self.model.classification_layers))],
                bias_sizes=[self.model.classification_layers[i].bias.shape for i in
                            range(len(self.model.classification_layers))],
                device=self.device)
        elif config.optimiser_type == 'nAdagrad':
            self.optimiser = NeuronIntegratorOptimiser(
                weight_sizes=[self.model.classification_layers[i].weight.shape for i in
                              range(len(self.model.classification_layers))],
                bias_sizes=[self.model.classification_layers[i].bias.shape for i in
                            range(len(self.model.classification_layers))],
                device=self.device)
        elif config.optimiser_type == 'fAdagrad':
            self.optimiser = NetworkIntegratorOptimiser(
                weight_sizes=[self.model.classification_layers[i].weight.shape for i in
                              range(len(self.model.classification_layers))],
                bias_sizes=[self.model.classification_layers[i].bias.shape for i in
                            range(len(self.model.classification_layers))],
                device=self.device)
        elif config.optimiser_type == 'Adam':
            self.optimiser = AdamOptimiser(
                weight_sizes=[self.model.classification_layers[i].weight.shape for i in
                              range(len(self.model.classification_layers))],
                bias_sizes=[self.model.classification_layers[i].bias.shape for i in
                            range(len(self.model.classification_layers))],
                device=self.device)
        elif config.optimiser_type == 'globalCost':
            self.optimiser = NetworkCostOptimiser(device=self.device)

    def update_model_weights(self, global_cost=None):
        self.model.update_weights(lr=self.lr, lr_Y=self.lr_Y, lr_pyr_intn=self.lr_pyr_intn,
                                  lr_intn_pyr=self.lr_intn_pyr, weight_decay=self.weight_decay,
                                  optimiser=self.optimiser, global_cost=global_cost)

    def log_state(self):
        super().log_state()
        state = dict()

        if self.config.log_mode == 'all' or self.config.log_mode == 'no_states':
            layer_angles_dict = self.get_layer_angles()
            layer_update_magnitudes_dict = self.get_layer_update_magnitudes()
            loggable_update_magnitudes_dict = {k: wandb.Histogram(np_histogram=(v, np.array(range(len(v) + 1)))) for
                                               k, v in
                                               layer_update_magnitudes_dict.items()}

            state = state | layer_angles_dict | loggable_update_magnitudes_dict

        wandb.log(state, commit=False)

    def log_layer_states(self):
        state = dict()

        if self.config.log_mode == 'all':
            layer_states_dict = self.get_layer_states()
            loggable_layer_states_dict = {k: wandb.Histogram(v) for k, v in layer_states_dict.items()}

            state = state | loggable_layer_states_dict

        wandb.log(state, commit=False)

    def get_layer_angles(self):
        W_Y_angles = self.model.weight_angles_W_Y()
        W_Y_angles_dict = {f'angle_W_Y ({i})': W_Y_angles[i] for i in range(len(W_Y_angles))}

        bp_angles = self.model.bp_angles()
        bp_angles_dict = {f'angle_bp ({i})': bp_angles[i] for i in range(len(bp_angles))}
        fa_angles = self.model.fa_angles()
        fa_angles_dict = {f'angle_fa ({i})': fa_angles[i] for i in range(len(fa_angles))}
        fa_to_bp_angles = self.model.fa_to_bp_angles()
        fa_to_bp_angles_dict = {f'angle_fa_to_bp ({i})': fa_to_bp_angles[i] for i in range(len(fa_to_bp_angles))}

        global_bp_angle = self.model.global_bp_angle()
        global_fa_angle = self.model.global_fa_angle()
        global_fa_to_bp_angle = self.model.global_fa_to_bp_angle()

        global_angles_dict = {'global_bp_angle': global_bp_angle,
                              'global_fa_angle': global_fa_angle,
                              'global_fa_to_bp_angle': global_fa_to_bp_angle}

        layer_angles_dict = W_Y_angles_dict | bp_angles_dict | fa_angles_dict | fa_to_bp_angles_dict | global_angles_dict
        return layer_angles_dict

    def get_layer_update_magnitudes(self):
        bp_grad_magnitudes = self.model.bp_grad_magnitudes()
        grad_magnitudes = self.model.grad_magnitudes()

        layer_update_magnitudes_dict = {'bp_grad_magnitudes': bp_grad_magnitudes,
                                        'grad_magnitudes': grad_magnitudes}
        return layer_update_magnitudes_dict

    def get_layer_states(self):
        layer_states_dict = dict()
        # Get hidden layer states

        return layer_states_dict

    def get_save_strings(self):
        save_strings = super().get_save_strings()

        return save_strings

    def save_training_setup(self):
        super().save_training_setup()
        run_directory = os.path.join(self.config.working_directory, 'results', self.config.run_name)
        shutil.copyfile(os.path.join(self.config.working_directory, 'model_trainers.py'),
                        os.path.join(run_directory, 'model_trainers.py'))
        shutil.copyfile(os.path.join(self.config.working_directory, "modules/networks_edn.py"),
                        os.path.join(run_directory, "networks_edn.py"))
        shutil.copyfile(os.path.join(self.config.working_directory, "modules/layers_edn.py"),
                        os.path.join(run_directory, "layers_edn.py"))


class ANNTrainer(BioModelTrainer):
    def __init__(self):
        super().__init__()

    def parse_model_params(self, parser):
        parser.add_argument("--p_baseline", type=float, help="Output layer baseline burst probability",
                            default=0.5)
        parser.add_argument("--n_hidden_layers", type=int,
                            help="Number of hidden layers",
                            default=3)
        parser.add_argument("--n_hidden_units", type=int,
                            help="Number of hidden units in each layer",
                            default=500)

        parser.add_argument('--Y_learning', default=False, help="Whether to update Y feedback weights",
                            type=lambda x: (str(x).lower() == 'true'))
        parser.add_argument('--Q_learning', default=False, help="Whether to update Y feedback weights",
                            type=lambda x: (str(x).lower() == 'true'))

        parser.add_argument("--lr", help="Learning rate for hidden layers", type=float, default=0.1)
        parser.add_argument("--Y_lr", type=float, default=0.001)
        parser.add_argument("--Q_lr", type=float, default=0.001)

        parser.add_argument("--momentum", type=float, help="Momentum", default=0.0)
        parser.add_argument("--weight_decay", type=float, help="Weight decay", default=0.0)

        parser.add_argument("--Y_mode", type=str, help="Must be 'tied', 'symmetric_init' or 'random_init",
                            default='random_init')
        parser.add_argument("--Y_scale", type=float, help="Scale of the feedback weights.", default=1.0)

        parser.add_argument("--Q_mode", type=str, help="Must be 'tied', 'symmetric_init' or 'random_init",
                            default='symmetric_init')
        parser.add_argument("--Q_scale", type=float, help="", default=1.0)

        parser.add_argument("--no_teaching_signal", default=False, type=lambda x: (str(x).lower() == 'true'))
        parser.add_argument("--feedforward_noise", default=None, type=float)

        parser.add_argument("--optimiser_type", type=str, help="Optimiser type", default='SGD')

        parser.add_argument("--use_conv", help="Use conv layers", default=False,
                            type=lambda x: (str(x).lower() == 'true'))

        model_args, _ = parser.parse_known_args()

        return model_args

    def set_config(self, config):
        self.config = config

        if config.use_conv:
            self.lr = [config.lr] * 5
            self.lr_Y = [config.Y_lr] * 4 + [None]
            self.lr_Q = [config.Q_lr] * 4 + [None]
        else:
            self.lr = [config.lr] * (config.n_hidden_layers + 1)
            self.lr_Y = [config.Y_lr] * config.n_hidden_layers + [None]
            self.lr_Q = [config.Q_lr] * config.n_hidden_layers + [None]

        self.momentum = config.momentum
        self.weight_decay = config.weight_decay

        self.batch_size = config.batch_size
        self.no_teaching_signal = config.no_teaching_signal
        self.feedforward_noise = config.feedforward_noise

        if config.use_conv:
            self.model = ConvBurstCCN(n_inputs=None, n_outputs=10, p_baseline=0.5, n_hidden_layers=None,
                                      n_hidden_units=None, Y_mode=config.Y_mode, Q_mode=config.Q_mode,
                                      Y_scale=config.Y_scale,
                                      Q_scale=config.Q_scale, Y_learning=config.Y_learning,
                                      Q_learning=config.Q_learning, device=self.device)
        else:
            self.model = BurstCCN(
                n_inputs=784,
                n_outputs=10,
                p_baseline=config.p_baseline,
                n_hidden_layers=config.n_hidden_layers,
                n_hidden_units=config.n_hidden_units,
                Y_learning=config.Y_learning,
                Y_mode=config.Y_mode,
                Y_scale=config.Y_scale,
                Q_learning=config.Q_learning,
                Q_mode=config.Q_mode,
                Q_scale=config.Q_scale,
                device=self.device).to(self.device)

        if config.optimiser_type == 'SGD':
            if self.momentum == 0.0:
                self.optimiser = SGDOptimiser()
            else:
                self.optimiser = SGDMomentumOptimiser(
                    weight_sizes=[self.model.classification_layers[i].weight.shape for i in
                                  range(len(self.model.classification_layers))],
                    bias_sizes=[self.model.classification_layers[i].bias.shape for i in
                                range(len(self.model.classification_layers))],
                    momentum=self.momentum,
                    device=self.device)
        elif config.optimiser_type == 'sAdagrad':
            self.optimiser = SynapseIntegratorOptimiser(
                weight_sizes=[self.model.classification_layers[i].weight.shape for i in
                              range(len(self.model.classification_layers))],
                bias_sizes=[self.model.classification_layers[i].bias.shape for i in
                            range(len(self.model.classification_layers))],
                device=self.device)
        elif config.optimiser_type == 'nAdagrad':
            self.optimiser = NeuronIntegratorOptimiser(
                weight_sizes=[self.model.classification_layers[i].weight.shape for i in
                              range(len(self.model.classification_layers))],
                bias_sizes=[self.model.classification_layers[i].bias.shape for i in
                            range(len(self.model.classification_layers))],
                device=self.device)
        elif config.optimiser_type == 'lAdagrad':
            self.optimiser = LayerIntegratorOptimiser(
                weight_sizes=[self.model.classification_layers[i].weight.shape for i in
                              range(len(self.model.classification_layers))],
                bias_sizes=[self.model.classification_layers[i].bias.shape for i in
                            range(len(self.model.classification_layers))],
                device=self.device)
        elif config.optimiser_type == 'fAdagrad':
            self.optimiser = NetworkIntegratorOptimiser(
                weight_sizes=[self.model.classification_layers[i].weight.shape for i in
                              range(len(self.model.classification_layers))],
                bias_sizes=[self.model.classification_layers[i].bias.shape for i in
                            range(len(self.model.classification_layers))],
                device=self.device)
        elif config.optimiser_type == 'Adam':
            self.optimiser = AdamOptimiser(
                weight_sizes=[self.model.classification_layers[i].weight.shape for i in
                              range(len(self.model.classification_layers))],
                bias_sizes=[self.model.classification_layers[i].bias.shape for i in
                            range(len(self.model.classification_layers))],
                device=self.device)
        elif config.optimiser_type == 'globalCost':
            self.optimiser = NetworkCostOptimiser(device=self.device)

    def update_model_weights(self, global_cost=None):
        if self.model.Y_mode == 'tied':
            self.model.update_weights(lrs=self.lr, lrs_Y=self.lr_Y, lrs_Q=self.lr_Q, optimiser=self.optimiser,
                                      global_cost=global_cost, weight_decay=self.weight_decay, use_backprop=True,
                                      use_feedback_alignment=False)
        else:
            self.model.update_weights(lrs=self.lr, lrs_Y=self.lr_Y, lrs_Q=self.lr_Q, optimiser=self.optimiser,
                                      global_cost=global_cost, weight_decay=self.weight_decay, use_backprop=False,
                                      use_feedback_alignment=True)

    def log_state(self):
        super().log_state()
        state = dict()

        if self.config.log_mode == 'all' or self.config.log_mode == 'no_states':
            layer_angles_dict = self.get_layer_angles()
            layer_update_magnitudes_dict = self.get_layer_update_magnitudes()
            loggable_update_magnitudes_dict = {k: wandb.Histogram(np_histogram=(v, np.array(range(len(v) + 1)))) for
                                               k, v in
                                               layer_update_magnitudes_dict.items()}

            state = state | layer_angles_dict | loggable_update_magnitudes_dict

        wandb.log(state, commit=False)

    def log_layer_states(self):
        state = dict()
        if self.config.log_mode == 'all':
            layer_states_dict = self.get_layer_states()
            loggable_layer_states_dict = {k: wandb.Histogram(v) for k, v in layer_states_dict.items()}

            state = state | loggable_layer_states_dict

        wandb.log(state, commit=False)

    def get_layer_angles(self):
        W_Y_angles = self.model.weight_angles_W_Y()
        W_Y_angles_dict = {f'angle_W_Y ({i})': W_Y_angles[i] for i in range(len(W_Y_angles))}

        bp_angles = self.model.bp_angles()
        bp_angles_dict = {f'angle_bp ({i})': bp_angles[i] for i in range(len(bp_angles))}
        fa_angles = self.model.fa_angles()
        fa_angles_dict = {f'angle_fa ({i})': fa_angles[i] for i in range(len(fa_angles))}
        fa_to_bp_angles = self.model.fa_to_bp_angles()
        fa_to_bp_angles_dict = {f'angle_fa_to_bp ({i})': fa_to_bp_angles[i] for i in range(len(fa_to_bp_angles))}

        global_bp_angle = self.model.global_bp_angle()
        global_fa_angle = self.model.global_fa_angle()
        global_fa_to_bp_angle = self.model.global_fa_to_bp_angle()

        global_angles_dict = {'global_bp_angle': global_bp_angle,
                              'global_fa_angle': global_fa_angle,
                              'global_fa_to_bp_angle': global_fa_to_bp_angle}

        layer_angles_dict = W_Y_angles_dict | bp_angles_dict | fa_angles_dict | fa_to_bp_angles_dict | global_angles_dict
        return layer_angles_dict

    def get_layer_update_magnitudes(self):
        bp_grad_magnitudes = self.model.bp_grad_magnitudes()
        grad_magnitudes = self.model.grad_magnitudes()

        layer_update_magnitudes_dict = {'bp_grad_magnitudes': bp_grad_magnitudes,
                                        'grad_magnitudes': grad_magnitudes}
        return layer_update_magnitudes_dict

    def get_layer_states(self):
        layer_states_dict = dict()
        # Get hidden layer states
        for i in range(len(self.model.classification_layers) - 1):
            layer = self.model.classification_layers[i]
            layer_states_dict[f"hidden{i + 1}.event_rate"] = layer.e.flatten().cpu().numpy()
            layer_states_dict[f"hidden{i + 1}.burst_prob"] = layer.p_t.flatten().cpu().numpy()
            layer_states_dict[f"hidden{i + 1}.burst_rate"] = layer.b_t.flatten().cpu().numpy()

        # Get output layer states
        output_layer = self.model.classification_layers[-1]
        layer_states_dict["output.event_rate"] = output_layer.e.flatten().cpu().numpy()
        layer_states_dict["output.burst_prob"] = output_layer.p_t.flatten().cpu().numpy()
        layer_states_dict["output.burst_rate"] = output_layer.b_t.flatten().cpu().numpy()

        return layer_states_dict

    def get_save_strings(self):
        save_strings = super().get_save_strings()

        save_strings.append(f"Baseline burst probability: {self.config.p_baseline}")

        save_strings.append(f"Q weight learning: {self.config.Q_learning}")
        save_strings.append(f"Q weight learning rate: {self.config.Q_lr}")
        save_strings.append(f"Q weight initialization standard deviation: {self.config.Q_scale}")

        return save_strings

    def save_training_setup(self):
        super().save_training_setup()
        run_directory = os.path.join(self.config.working_directory, 'results', self.config.run_name)
        shutil.copyfile(os.path.join(self.config.working_directory, 'model_trainers.py'),
                        os.path.join(run_directory, 'model_trainers.py'))
        shutil.copyfile(os.path.join(self.config.working_directory, "modules/networks_burstccn.py"),
                        os.path.join(run_directory, "networks_burstccn.py"))
        shutil.copyfile(os.path.join(self.config.working_directory, "modules/layers_burstccn.py"),
                        os.path.join(run_directory, "layers_burstccn.py"))

# class ANNTrainer(BioModelTrainer):
#     def __init__(self):
#         super().__init__()
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#         # Overwritten by parse_model_params
#         self.model = None
#
#         self.log_frequency = 200
#
#     def parse_model_params(self, parser):
#
#         parser.add_argument("--lr", help="Learning rate for hidden layers", type=float, default=0.1)
#         parser.add_argument("--n_hidden_layers", type=int,
#                             help="Number of hidden layers",
#                             default=3)
#         parser.add_argument("--n_hidden_units", type=int,
#                             help="Number of hidden units in each layer",
#                             default=500)
#
#         parser.add_argument("--momentum", type=float, help="Momentum", default=0.0)
#         parser.add_argument("--weight_decay", type=float, help="Weight decay", default=0.0)
#
#         parser.add_argument("--Y_mode", type=str, help="Must be 'tied', 'symmetric_init' or 'random_init",
#                             default='random_init')
#         parser.add_argument("--Y_scale", type=float, help="Scale of the feedback weights.", default=1.0)
#
#         model_args, _ = parser.parse_known_args()
#
#         return model_args
#
#     def set_config(self, config):
#         self.config = config
#
#         self.lr = [config.lr] * (config.n_hidden_layers + 1)
#         self.momentum = config.momentum
#         self.weight_decay = config.weight_decay
#         self.batch_size = config.batch_size
#
#         self.model = MNISTNetFA(n_hidden_layers=config.n_hidden_layers, n_hidden_units=config.n_hidden_units,
#                                 Y_mode=config.Y_mode, Y_scale=config.Y_scale, device=self.device).to(
#             self.device)
#
#         if config.optimiser_type == 'SGD':
#             if self.momentum == 0.0:
#                 self.optimiser = SGDOptimiser()
#             else:
#                 self.optimiser = SGDMomentumOptimiser(
#                     weight_sizes=[self.model.classification_layers[i].weight.shape for i in
#                                   range(len(self.model.classification_layers))],
#                     bias_sizes=[self.model.classification_layers[i].bias.shape for i in
#                                 range(len(self.model.classification_layers))],
#                     momentum=self.momentum,
#                     device=self.device)
#         elif config.optimiser_type == 'sAdagrad':
#             self.optimiser = SynapseIntegratorOptimiser(
#                 weight_sizes=[self.model.classification_layers[i].weight.shape for i in
#                               range(len(self.model.classification_layers))],
#                 bias_sizes=[self.model.classification_layers[i].bias.shape for i in
#                             range(len(self.model.classification_layers))],
#                 device=self.device)
#         elif config.optimiser_type == 'nAdagrad':
#             self.optimiser = NeuronIntegratorOptimiser(
#                 weight_sizes=[self.model.classification_layers[i].weight.shape for i in
#                               range(len(self.model.classification_layers))],
#                 bias_sizes=[self.model.classification_layers[i].bias.shape for i in
#                             range(len(self.model.classification_layers))],
#                 device=self.device)
#         elif config.optimiser_type == 'fAdagrad':
#             self.optimiser = NetworkIntegratorOptimiser(
#                 weight_sizes=[self.model.classification_layers[i].weight.shape for i in
#                               range(len(self.model.classification_layers))],
#                 bias_sizes=[self.model.classification_layers[i].bias.shape for i in
#                             range(len(self.model.classification_layers))],
#                 device=self.device)
#         elif config.optimiser_type == 'Adam':
#             self.optimiser = AdamOptimiser(
#                 weight_sizes=[self.model.classification_layers[i].weight.shape for i in
#                               range(len(self.model.classification_layers))],
#                 bias_sizes=[self.model.classification_layers[i].bias.shape for i in
#                             range(len(self.model.classification_layers))],
#                 device=self.device)
#         elif config.optimiser_type == 'globalCost':
#             self.optimiser = NetworkCostOptimiser(device=self.device)
#
#     def update_model_weights(self, global_cost=None):
#         self.model.update_weights(lrs=self.lr, momentum=self.momentum, weight_decay=self.weight_decay,
#                                   global_cost=global_cost,
#                                   optimiser=self.optimiser, batch_size=self.batch_size)
#
#     def log_state(self):
#         pass
#         # layer_update_magnitudes_dict = self.get_layer_update_magnitudes()
#         #
#         # loggable_layer_states_dict = {k: wandb.Histogram(v) for k, v in layer_states_dict.items()}
#         # loggable_update_magnitudes_dict = {k: wandb.Histogram(np_histogram=(v, np.array(range(len(v) + 1)))) for k, v in
#         #                                    layer_update_magnitudes_dict.items()}
#         #
#         # wandb.log(layer_angles_dict | loggable_layer_states_dict | loggable_update_magnitudes_dict, commit=False)
#
#     def get_layer_angles(self):
#         pass
#         # W_Y_angles = self.model.weight_angles_W_Y()
#         # W_Y_angles_dict = {f'angle_W_Y ({i})': W_Y_angles[i] for i in range(len(W_Y_angles))}
#         #
#         # layer_angles_dict = W_Y_angles_dict
#         # return layer_angles_dict
#
#     def get_layer_update_magnitudes(self):
#         pass
#         # grad_magnitudes = self.model.grad_magnitudes()
#         #
#         # layer_update_magnitudes_dict = {'grad_magnitudes': grad_magnitudes}
#         # return layer_update_magnitudes_dict
#
#     def get_layer_states(self):
#         layer_states_dict = dict()
#
#         return layer_states_dict
#
#     def get_save_strings(self):
#         save_strings = []
#
#         # save a human-readable text file containing simulation details
#         timestamp = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
#         save_strings.append(f"Simulation run @ {timestamp}")
#         save_strings.append(f"Number of epochs: {self.config.n_epochs}")
#         save_strings.append(f"Batch size: {self.config.batch_size}")
#         save_strings.append(f"Using validation set: {self.config.use_validation}")
#
#         save_strings.append(f"Feedforward learning rates: {self.config.lr}")
#         save_strings.append(f"Feedback weight initialization scale: {self.config.Y_scale}")
#         save_strings.append(f"Momentum: {self.config.momentum}")
#         save_strings.append(f"Weight decay: {self.config.weight_decay}")
#
#         return save_strings
#
#     def save_training_setup(self):
#         run_directory = os.path.join(self.config.working_directory, 'results', self.config.run_name)
#
#         if not os.path.exists(run_directory):
#             os.makedirs(run_directory)
#
#         save_strings = self.get_save_strings()
#
#         with open(os.path.join(run_directory, "params.txt"), "w") as f:
#             for save_string in save_strings:
#                 f.write(save_string + "\n")
