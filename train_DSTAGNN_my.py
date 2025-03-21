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
from lib.utils1 import load_graphdata_channel1, get_adjacency_matrix2, compute_val_loss_mstgcn, predict_and_save_results_mstgcn
from tensorboardX import SummaryWriter


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # To ensure reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='configurations/Gambia.conf', type=str,
                        help="configuration file path")
    args = parser.parse_args()

    # Read configuration file
    config = configparser.ConfigParser()
    print('Read configuration file: %s' % (args.config))
    config.read(args.config)
    data_config = config['Data']
    training_config = config['Training']

    # Data configuration
    adj_filename = data_config['adj_filename']
    graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
    stag_filename = data_config.get('stag_filename', None)  # Optional: Handle missing STAG file
    strg_filename = data_config.get('strg_filename', None)  # Optional: Handle missing STRG file
    num_of_vertices = int(data_config['num_of_vertices'])
    points_per_hour = int(data_config['points_per_hour'])
    num_for_predict = int(data_config['num_for_predict'])
    len_input = int(data_config['len_input'])
    dataset_name = data_config['dataset_name']

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

    # Set device (GPU or CPU)
    if ctx == 'cpu':
        DEVICE = torch.device('cpu')
    else:
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("DEVICE:", DEVICE)

    # Create output directory
    folder_dir = '%s_h%dd%dw%d_channel%d_%e' % (model_name, num_of_hours, num_of_days, num_of_weeks, in_channels, learning_rate)
    print('folder_dir:', folder_dir)
    params_path = os.path.join('myexperiments', dataset_name, folder_dir)
    print('params_path:', params_path)

    if not os.path.exists(params_path):
        os.makedirs(params_path)
        print('create params directory %s' % (params_path))

    # Load data
    _, train_loader, train_target_tensor, _, val_loader, val_target_tensor, _, test_loader, test_target_tensor, _mean, _std = load_graphdata_channel1(
        graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size)

    # Load adjacency matrix
    if dataset_name in ['PEMS04', 'PEMS08', 'PEMS07', 'PEMS03']:
        adj_mx = get_adjacency_matrix2(adj_filename, num_of_vertices)
    else:
        adj_mx = load_weighted_adjacency_matrix2(adj_filename, num_of_vertices)

    # Handle STAG and STRG files (set to None if not provided)
    adj_TMD = None
    adj_pa = None
    if stag_filename is not None and stag_filename != 'None':
        adj_TMD = load_weighted_adjacency_matrix(stag_filename, num_of_vertices)
    if strg_filename is not None and strg_filename != 'None':
        adj_pa = load_PA(strg_filename)

    # Create the model
    net = make_model(DEVICE, num_of_d, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, adj_mx,
                     adj_pa, adj_TMD, num_for_predict, len_input, num_of_vertices, d_model, d_k, d_v, n_heads)
    net = net.to(DEVICE)  # Move model to the correct device

    # Initialize SummaryWriter
    sw = SummaryWriter(logdir=params_path, flush_secs=5)

    # Train the model
    criterion = nn.SmoothL1Loss().to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

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
            encoder_inputs = encoder_inputs.to(DEVICE, non_blocking=True)  # Move inputs to the correct device
            labels = labels.to(DEVICE, non_blocking=True)  # Move labels to the correct device

            optimizer.zero_grad()
            outputs = net(encoder_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            training_loss = loss.item()
            global_step += 1
            sw.add_scalar('training_loss', training_loss, global_step)

            if global_step % 1000 == 0:
                print('global step: %s, training loss: %.2f, time: %.2fs' % (global_step, training_loss, time() - start_time))

    print('best epoch:', best_epoch)
    # Apply the best model on the test set
    predict_main(best_epoch, test_loader, test_target_tensor, _mean, _std, 'test', DEVICE)


def predict_main(global_step, data_loader, data_target_tensor, _mean, _std, type, DEVICE):
    '''
    :param global_step: int
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param mean: (1, 1, 3, 1)
    :param std: (1, 1, 3, 1)
    :param type: string
    :param DEVICE: torch.device, device to use for inference
    :return:
    '''
    params_filename = os.path.join(params_path, 'epoch_%s.params' % global_step)
    print('load weight from:', params_filename)

    # Load the model state dict
    state_dict = torch.load(params_filename, map_location=DEVICE)
    net.load_state_dict(state_dict)

    # Set the model to evaluation mode
    net.eval()

    # Perform inference
    with torch.no_grad():
        predictions = []
        for batch_index, batch_data in enumerate(data_loader):
            encoder_inputs, _ = batch_data
            encoder_inputs = encoder_inputs.to(DEVICE, non_blocking=True)  # Move inputs to the correct device

            # Forward pass
            outputs = net(encoder_inputs)
            predictions.append(outputs.cpu())

        # Concatenate predictions
        predictions = torch.cat(predictions, dim=0)

    # Save results
    predict_and_save_results_mstgcn(net, data_loader, data_target_tensor, global_step, _mean, _std, params_path, type)


if __name__ == "__main__":
    seed_torch(1)  # Set seed for reproducibility
    train_main()
