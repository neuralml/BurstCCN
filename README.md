# Single-phase deep learning in cortico-cortical networks

This repository contains the code to run the BurstCCN, Burstprop and EDN models to reproduce the results in "Single-phase deep learning in cortico-cortical networks" available on [arXiv](https://arxiv.org/pdf/2206.11769).

## Requirements

To install the requrired packages, create a new conda environment using:

```
conda create --name <env> --file requirements.txt
```

## Training the discrete-time model


### Command line

To train the burstccn model on MNIST, first set up a [wandb](wandb.ai) project and then simply use the command:

```
python train_model.py --run_name=RUN_NAME 
                      --wandb_project=WANDB_PROJECT 
                      --wandb_entity=WANDB_ENTITY 
                      --model_type=burstccn 
                      --dataset=mnist 
                      [--n_epochs=N_EPOCHS]
                      [--batch_size=BATCH_SIZE]
                      [--p_baseline=P_BASELINE] 
                      [--n_hidden_layers=N_HIDDEN_LAYERS] 
                      [--n_hidden_units=N_HIDDEN_UNITS] 
                      [--lr=LEARNING_RATE] 
                      [--Y_lr=Y_LEARNING_RATE] 
                      [--Q_lr=Q_LEARNING_RATE] 
                      [--momentum=MOMENTUM] 
                      [--weight_decay=WEIGHT_DECAY]
                      [--Y_learning=Y_LEARNING] 
                      [--Y_mode=Y_MODE] 
                      [--Y_scale=Y_SCALE] 
                      [--Q_learning=Q_LEARNING] 
                      [--Q_mode=Q_MODE] 
                      [--Q_scale=Q_SCALE] 
```

|Parameter name| Description |
| --- | --- |
| run_name | Name of the run to use on wandb when logging the results (required). |
| wandb_project | Project name on wandb (required). |
| wandb_entity | Entity (user or group) name on wandb (required).|
| model_type | Type of model to train. Options=[ *ann* \| *burstccn* \| *burstprop* \| *edn* ] (default=burstccn). |
| dataset | Dataset to train model on. Options=[ *mnist* \| *cifar10* ] (default=mnist). |
| n_epochs | Number of epochs to train on (default=250). |
| batch_size | Size of batches to train on (default=32). |
| p_baseline | (**BurstCCN only**) Baseline burst probability (default=0.5). |
| n_hidden_layers | Number of hidden layers to use in the model (default=3). |
| n_hidden_units | Number of units to use in each hidden layer (default=500). |
| lr | Learning rate of feedforward weights (default=0.1). |
| Y_lr | Learning rate of feedback weights (default=0.0). |
| Q_lr | (**BurstCCN only**) Learning rate of feedback weights (default=0.0). |
| momentum | Momentum value in [0,1] (default=0.0). |
| weight_decay | Weight decay value (default=0.0) |
| Y_learning | Whether to learn feedback Y weights, [ *True* \| *False* ] (default=False) |
| Y_mode | Y weight regime. Options=[ *random* \| *symmetric_init* \| *tied* ] (see [jupyter notebook](Training%20BurstCCN%20on%20MNIST.ipynb) for an explanation, default=random_init) |
| Y_scale | Scale of the Y weights. Is either the standard deviation of the initialisation or scale relative to feedforward weights depending on the Y_mode. (see [jupyter notebook](Training%20BurstCCN%20on%20MNIST.ipynb) for an explanation, default=1.0) |
| Q_learning | (**BurstCCN only**) Whether to learn feedback Q weights, [ *True* \| *False* ] (default=False) |
| Q_mode | (**BurstCCN only**) Q weight regime. Options=[ *random* \| *symmetric_init* \| *tied* ] (see [jupyter notebook](Training%20BurstCCN%20on%20MNIST.ipynb) for an explanation, default=symmetric_init) |
| Q_scale | (**BurstCCN only**) Scale of the Q weights. Is either the standard deviation of the initialisation or scale relative to feedback Y weights depending on the Q_mode. (see [jupyter notebook](Training%20BurstCCN%20on%20MNIST.ipynb) for an explanation, default=1.0) |

### Jupyter notebook example

Run the jupyter notebook using the command:

```
jupyter notebook "Training BurstCCN on MNIST.ipynb"
```


## Training the continuous-time model

To train the continuous-time burstccn on a non-linear regression task, first set up a [wandb](wandb.ai) project and then simply use the command:

```
python train_continuous_model.py --run_name=RUN_NAME 
                      --wandb_project=WANDB_PROJECT 
                      --wandb_entity=WANDB_ENTITY 
                      --model_type=burstccn
```