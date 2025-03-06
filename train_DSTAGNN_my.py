#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
from time import time
import shutil
import argparse
import configparser
from model.DSTAGNN_my import make_model
from lib.dataloader import load_weighted_adjacency_matrix, load_weighted_adjacency_matrix2, load_PA
from lib.utils1 import get_adjacency_matrix2, compute_val_loss_mstgcn, predict_and_save_results_mstgcn
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.distributed as dist


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # To ensure reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'  # Change port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class ChunkedDataset(Dataset):
    def __init__(self, data_file, chunk_size, num_nodes, num_features, num_timesteps, mode='train'):
        self.data_file = data_file
        self.chunk_size = chunk_size
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_timesteps = num_timesteps
        self.mode = mode

        # Load metadata (e.g., mean and std)
        with np.load(data_file) as data:
            self.mean = data['mean']
            self.std = data['std']

    def __len__(self):
        return self.num_timesteps // self.chunk_size

    def __getitem__(self, idx):
        # Load a chunk of data
        start_idx = idx * self.chunk_size
        end_idx = start_idx + self.chunk_size

        with np.load(self.data_file) as data:
            if self.mode == 'train':
                x = data['train_x'][start_idx:end_idx]  # Shape: (chunk_size, num_nodes, num_features, 1)
                y = data['train_target'][start_idx:end_idx]  # Shape: (chunk_size, num_nodes)
            elif self.mode == 'val':
                x = data['val_x'][start_idx:end_idx]
                y = data['val_target'][start_idx:end_idx]
            elif self.mode == 'test':
                x = data['test_x'][start_idx:end_idx]
                y = data['test_target'][start_idx:end_idx]

        # Normalize the data
        x = (x - self.mean) / self.std

        # Convert to PyTorch tensors
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)

        return x, y


def load_graphdata_channel1(data_file, num_hours, num_days, num_weeks, device, batch_size, chunk_size):
    # Load the .npz file
    data = np.load(data_file)

    # Extract the required arrays
    train_x = data['train_x']  # Shape: (num_samples, num_nodes, num_features, num_timesteps)
    train_target = data['train_target']  # Shape: (num_samples, num_nodes)
    val_x = data['val_x']  # Shape: (num_samples, num_nodes, num_features, num_timesteps)
    val_target = data['val_target']  # Shape: (num_samples, num_nodes)
    test_x = data['test_x']  # Shape: (num_samples, num_nodes, num_features, num_timesteps)
    test_target = data['test_target']  # Shape: (num_samples, num_nodes)
    mean = data['mean']  # Shape: (1, 1, num_features, 1)
    std = data['std']  # Shape: (1, 1, num_features, 1)

    # Convert to PyTorch tensors and move to the specified device
    train_x = torch.FloatTensor(train_x).to(device)
    train_target = torch.FloatTensor(train_target).to(device)
    val_x = torch.FloatTensor(val_x).to(device)
    val_target = torch.FloatTensor(val_target).to(device)
    test_x = torch.FloatTensor(test_x).to(device)
    test_target = torch.FloatTensor(test_target).to(device)
    mean = torch.FloatTensor(mean).to(device)
    std = torch.FloatTensor(std).to(device)

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(train_x, train_target)
    val_dataset = torch.utils.data.TensorDataset(val_x, val_target)
    test_dataset = torch.utils.data.TensorDataset(test_x, test_target)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    return train_loader, val_loader, test_loader, mean, std

def train_main(rank, world_size, args):
    setup(rank, world_size)

    # Read configuration file
    config = configparser.ConfigParser()
    print('Read configuration file: %s' % (args.config))
    config.read(args.config)
    data_config = config['Data']
    training_config = config['Training']

    # Data configuration
    adj_filename = data_config['adj_filename']
    graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
    num_of_vertices = int(data_config['num_of_vertices'])
    points_per_hour = int(data_config['points_per_hour'])
    num_for_predict = int(data_config['num_for_predict'])
    len_input = int(data_config['len_input'])
    dataset_name = data_config['dataset_name']
    chunk_size = int(data_config.get('chunk_size', 1000))  # Default chunk size: 1000

    # Training configuration
    model_name = training_config['model_name']
    graph_use = training_config['graph']
    ctx = training_config['ctx']
    learning_rate = float(training_config['learning_rate'])
    epochs = int(training_config['epochs'])
    start_epoch = int(training_config['start_epoch'])
    batch_size = int(training_config['batch_size'])
    num_of_weeks = int(training_config['num_of_weeks'])
    num_of_days = int(training_config['num_of_days'])
    num_of_hours = int(training_config['num_of_hours'])
    time_strides = 1
    d_model = int(training_config['d_model'])
    nb_chev_filter = int(training_config['nb_chev_filter'])
    nb_time_filter = int(training_config['nb_time_filter'])
    in_channels = int(training_config['in_channels'])
    num_of_d = in_channels
    nb_block = int(training_config['nb_block'])
    K = int(training_config['K'])
    n_heads = int(training_config['n_heads'])
    d_k = int(training_config['d_k'])
    d_v = d_k

    # Create output directory
    folder_dir = '%s_h%dd%dw%d_channel%d_%e' % (model_name, num_of_hours, num_of_days, num_of_weeks, in_channels, learning_rate)
    print('folder_dir:', folder_dir)
    params_path = os.path.join('myexperiments', dataset_name, folder_dir)
    print('params_path:', params_path)

    # Load data in chunks
    train_loader, val_loader, test_loader, mean, std = load_graphdata_channel1(
        graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, rank, batch_size, chunk_size)

    # Load adjacency matrix
    if dataset_name == 'PEMS04' or 'PEMS08' or 'PEMS07' or 'PEMS03':
        adj_mx = get_adjacency_matrix2(adj_filename, num_of_vertices)
    else:
        adj_mx = load_weighted_adjacency_matrix2(adj_filename, num_of_vertices)

    # Create the model
    net = make_model(rank, num_of_d, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, adj_mx,
                     None, None, num_for_predict, len_input, num_of_vertices, d_model, d_k, d_v, n_heads)

    # Wrap the model with DDP
    net = DDP(net, device_ids=[rank])

    # Use mixed precision training
    scaler = GradScaler()

    # Train the model
    criterion = nn.SmoothL1Loss().to(rank)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    sw = SummaryWriter(logdir=params_path, flush_secs=5)

    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf

    for epoch in range(start_epoch, epochs):
        print('current epoch: ', epoch)
        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)

        val_loss = compute_val_loss_mstgcn(net, val_loader, criterion, sw, epoch)
        print('val loss', val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(net.state_dict(), params_filename)
            print('best epoch: ', best_epoch)
            print('best val loss: ', best_val_loss)
            print('save parameters to file: %s' % params_filename)

        net.train()  # Ensure dropout layers are in train mode

        for batch_index, batch_data in enumerate(train_loader):
            encoder_inputs, labels = batch_data
            encoder_inputs = encoder_inputs.to(rank, non_blocking=True)
            labels = labels.to(rank, non_blocking=True)

            # Forward pass with mixed precision
            with autocast():
                outputs = net(encoder_inputs)
                loss = criterion(outputs, labels)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            if (batch_index + 1) % 8 == 0:  # Gradient accumulation
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            training_loss = loss.item()
            global_step += 1
            sw.add_scalar('training_loss', training_loss, global_step)

            if global_step % 1000 == 0:
                print('global step: %s, training loss: %.2f, time: %.2fs' % (global_step, training_loss, time() - start_time))

    print('best epoch:', best_epoch)
    # Apply the best model on the test set
    predict_main(best_epoch, test_loader, test_target, _mean, _std, 'test', rank)

    cleanup()

def predict_main(global_step, data_loader, data_target_tensor, _mean, _std, type, rank):
    '''
    :param global_step: int
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param mean: (1, 1, 3, 1)
    :param std: (1, 1, 3, 1)
    :param type: string
    :param rank: int, rank of the current process
    :return:
    '''
    params_filename = os.path.join(params_path, 'epoch_%s.params' % global_step)
    print('load weight from:', params_filename)

    # Load the model state dict
    state_dict = torch.load(params_filename, map_location=f'cuda:{rank}')
    net.load_state_dict(state_dict)

    # Set the model to evaluation mode
    net.eval()

    # Perform inference
    with torch.no_grad():
        predictions = []
        for batch_index, batch_data in enumerate(data_loader):
            encoder_inputs, _ = batch_data
            encoder_inputs = encoder_inputs.to(rank, non_blocking=True)

            # Forward pass with mixed precision
            with autocast():
                outputs = net(encoder_inputs)

            predictions.append(outputs.cpu())

        # Concatenate predictions
        predictions = torch.cat(predictions, dim=0)

    # Save results
    predict_and_save_results_mstgcn(net, data_loader, data_target_tensor, global_step, _mean, _std, params_path, type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='configurations/Gambia.conf', type=str,
                        help="configuration file path")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train_main, args=(world_size, args), nprocs=world_size, join=True)
