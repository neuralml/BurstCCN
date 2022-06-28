from abc import abstractmethod, ABC

import datetime
import os
import random
import argparse

import math

import numpy as np

import torch
import torch.nn.functional as F

from tqdm import tqdm
import wandb

from datasets import get_mnist_dataset, get_cifar10_dataset

from model_trainers import BurstPropTrainer, BurstCCNTrainer, EDNTrainer, ANNTrainer


def create_model_trainer(model_type):
    if model_type == 'burstprop':
        model_trainer = BurstPropTrainer()
    elif model_type == 'burstccn':
        model_trainer = BurstCCNTrainer()
    elif model_type == 'edn':
        model_trainer = EDNTrainer()
    elif model_type == 'ann':
        model_trainer = ANNTrainer()
    else:
        raise NotImplementedError()

    return model_trainer


def train(parser=None):
    if wandb.config.require_gpu and not torch.cuda.is_available():
        print("GPU not available and require_gpu is True!")
        return
    # model_trainer = create_model_trainer(config.model_type)

    model_trainer = create_model_trainer(wandb.config.model_type)

    if parser is None:
        parser = argparse.ArgumentParser()

    model_args = model_trainer.parse_model_params(parser)

    # config = argparse.Namespace(**(vars(model_args) | vars(config)))
    wandb.config.update(vars(model_args) | dict(wandb.config))

    torch.manual_seed(wandb.config.seed)
    torch.cuda.manual_seed(wandb.config.seed)
    torch.cuda.manual_seed_all(wandb.config.seed)
    random.seed(wandb.config.seed)
    np.random.seed(wandb.config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model_trainer.set_config(wandb.config)

    data_dir = os.path.join(wandb.config.working_directory, 'Data')
    if wandb.config.dataset == 'mnist':
        train_data_loader, validation_data_loader, test_data_loader = get_mnist_dataset(data_dir, train_batch_size=wandb.config.batch_size, use_validation=wandb.config.use_validation)
    elif wandb.config.dataset == 'cifar10':
        # train_data_loader, validation_data_loader, test_data_loader = get_mnist_dataset(data_dir,
        #                                                                                 train_batch_size=run_args.batch_size,
        #                                                                                 use_validation=run_args.use_validation)

        train_data_loader, validation_data_loader, test_data_loader = get_cifar10_dataset(data_dir,
                                                                                          train_batch_size=wandb.config.batch_size,
                                                                                          use_validation=wandb.config.use_validation)
    else:
        raise NotImplementedError()

    model_trainer.save_training_setup()

    test_error, test_loss = model_trainer.test(test_data_loader)

    best_test_error = test_error
    best_test_error_epoch = 0.0

    if wandb.config.use_validation:
        val_error, val_loss = model_trainer.test(test_data_loader)

        best_val_error = val_error
        best_val_error_epoch = 0.0

        wandb.log({'test_error': test_error,
                   'test_loss': test_loss,
                   'best_test_error': best_test_error,
                   'best_test_error_epoch': best_test_error_epoch,
                   'val_error': val_error,
                   'val_loss': val_loss,
                   'best_val_error': best_val_error,
                   'best_val_error_epoch': best_val_error_epoch})
    else:
        wandb.log({'test_error': test_error,
                   'test_loss': test_loss,
                   'best_test_error': best_test_error,
                   'best_test_error_epoch': best_test_error_epoch})

    for epoch in range(1, wandb.config.n_epochs + 1):
        print(f"\nEpoch {epoch}.")
        train_error, train_loss = model_trainer.train(train_data_loader)
        test_error, test_loss = model_trainer.test(test_data_loader)
        if wandb.config.use_validation:
            val_error, val_loss = model_trainer.test(validation_data_loader)

        # If model evaluation breaks then stop training
        if math.isnan(test_error):
            break

        if test_error < best_test_error:
            best_test_error = test_error
            best_test_error_epoch = epoch

        if wandb.config.max_stagnant_epochs is not None and epoch > best_test_error_epoch + wandb.config.max_stagnant_epochs:
            print(f"Test error has not improved for {wandb.config.max_stagnant_epochs} epochs. Stopping...")
            break

        if wandb.config.use_validation:
            if val_error < best_val_error:
                best_val_error = val_error
                best_val_error_epoch = epoch

            wandb.log({'train_error': train_error,
                       'train_loss': train_loss,
                       'test_error': test_error,
                       'test_loss': test_loss,
                       'best_test_error': best_test_error,
                       'best_test_error_epoch': best_test_error_epoch,
                       'val_error': val_error,
                       'val_loss': val_loss,
                       'best_val_error': best_val_error,
                       'best_val_error_epoch': best_val_error_epoch,
                       'epoch': epoch})

        else:
            wandb.log({'train_error': train_error,
                       'train_loss': train_loss,
                       'test_error': test_error,
                       'test_loss': test_loss,
                       'best_test_error': best_test_error,
                       'best_test_error_epoch': best_test_error_epoch,
                       'epoch': epoch})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, help='Name of the run', required=True)
    parser.add_argument('--model_type', type=str, help='Type of model to use', required=True)

    parser.add_argument('--wandb_project', type=str, help='Wandb project name.', required=True)
    parser.add_argument('--wandb_entity', type=str, help='Wandb entity name.', required=True)

    parser.add_argument('--seed', type=int, help='The seed number', default=1)
    parser.add_argument('--working_directory', type=str, default=os.getcwd())
    parser.add_argument('--dataset', type=str, default='mnist')

    parser.add_argument('--n_epochs', type=int, help='Number of epochs', default=250)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=32)
    parser.add_argument('--use_validation', default=False, help='Whether to the validation set',
                        type=lambda x: (str(x).lower() == 'true'))

    parser.add_argument('--require_gpu', default=False, type=lambda x: (str(x).lower() == 'true'))

    parser.add_argument('--log_mode', type=str, default='all')
    parser.add_argument('--max_stagnant_epochs', type=int, help='Number of epochs to run for with no improvement', default=None)

    run_args, _ = parser.parse_known_args()

    #wandb_project = 'burstccn'
    #wandb_entity = 'burstyboys'

    wandb.init(project=run_args.wandb_project, entity=run_args.wandb_entity, name=run_args.run_name, config=run_args)

    train(parser)