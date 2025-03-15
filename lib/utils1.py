import os
import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from .metrics import masked_mape_np
from scipy.sparse.linalg import eigs
from scipy.linalg import eigvalsh
from scipy.linalg import fractional_matrix_power

# keshihua
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker

def re_normalization(x, mean, std):
    x = x * std + mean
    return x


def max_min_normalization(x, _max, _min):
    x = 1. * (x - _min)/(_max - _min)
    x = x * 2. - 1.
    return x


def re_max_min_normalization(x, _max, _min):
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x


def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)

        if id_filename:

            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA

        else:

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance
            return A, distaneA

def get_adjacency_matrix2(distance_df_filename, num_of_vertices,
                         type_='connectivity', id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    type_: str, {connectivity, distance}

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    import csv

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx
                       for idx, i in enumerate(f.read().strip().split('\n'))}
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                A[id_dict[j], id_dict[i]] = 1
        return A

    # Fills cells in the matrix with distances.
    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            if type_ == 'connectivity':
                A[i, j] = 1
                # A[j, i] = 1
            elif type == 'distance':
                A[i, j] = 1 / distance
                A[j, i] = 1 / distance
            else:
                raise ValueError("type_ error, must be "
                                 "connectivity or distance!")
    return A


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


def calculate_laplacian_matrix(adj_mat, mat_type):
    n_vertex = adj_mat.shape[0]
    id_mat = np.asmatrix(np.identity(n_vertex))

    # D_row
    deg_mat_row = np.asmatrix(np.diag(np.sum(adj_mat, axis=1)))
    # D_com
    #deg_mat_col = np.asmatrix(np.diag(np.sum(adj_mat, axis=0)))

    # D = D_row as default
    deg_mat = deg_mat_row
    adj_mat = np.asmatrix(adj_mat)

    # wid_A = A + I
    wid_adj_mat = adj_mat + id_mat
    # wid_D = D + I
    wid_deg_mat = deg_mat + id_mat

    # Combinatorial Laplacian
    # L_com = D - A
    com_lap_mat = deg_mat - adj_mat

    if mat_type == 'id_mat':
        return id_mat
    elif mat_type == 'com_lap_mat':
        return com_lap_mat

    if (mat_type == 'sym_normd_lap_mat') or (mat_type == 'wid_sym_normd_lap_mat') or (mat_type == 'hat_sym_normd_lap_mat'):
        deg_mat_inv_sqrt = fractional_matrix_power(deg_mat, -0.5)
        deg_mat_inv_sqrt[np.isinf(deg_mat_inv_sqrt)] = 0.

        wid_deg_mat_inv_sqrt = fractional_matrix_power(wid_deg_mat, -0.5)
        wid_deg_mat_inv_sqrt[np.isinf(wid_deg_mat_inv_sqrt)] = 0.

        # Symmetric normalized Laplacian
        # For SpectraConv
        # To [0, 1]
        # L_sym = D^{-0.5} * L_com * D^{-0.5} = I - D^{-0.5} * A * D^{-0.5}
        sym_normd_lap_mat = id_mat - np.matmul(np.matmul(deg_mat_inv_sqrt, adj_mat), deg_mat_inv_sqrt)

        # For ChebConv
        # From [0, 1] to [-1, 1]
        # wid_L_sym = 2 * L_sym / lambda_max_sym - I
        #sym_max_lambda = max(np.linalg.eigvalsh(sym_normd_lap_mat))
        sym_max_lambda = max(eigvalsh(sym_normd_lap_mat))
        wid_sym_normd_lap_mat = 2 * sym_normd_lap_mat / sym_max_lambda - id_mat

        # For GCNConv
        # hat_L_sym = wid_D^{-0.5} * wid_A * wid_D^{-0.5}
        hat_sym_normd_lap_mat = np.matmul(np.matmul(wid_deg_mat_inv_sqrt, wid_adj_mat), wid_deg_mat_inv_sqrt)

        if mat_type == 'sym_normd_lap_mat':
            return sym_normd_lap_mat
        elif mat_type == 'wid_sym_normd_lap_mat':
            return wid_sym_normd_lap_mat
        elif mat_type == 'hat_sym_normd_lap_mat':
            return hat_sym_normd_lap_mat

    elif (mat_type == 'rw_normd_lap_mat') or (mat_type == 'wid_rw_normd_lap_mat') or (mat_type == 'hat_rw_normd_lap_mat'):
        try:
            # There is a small possibility that the degree matrix is a singular matrix.
            deg_mat_inv = np.linalg.inv(deg_mat)
        except:
            print(f'The degree matrix is a singular matrix. Cannot use random walk normalized Laplacian matrix.')
        else:
            deg_mat_inv[np.isinf(deg_mat_inv)] = 0.

        wid_deg_mat_inv = np.linalg.inv(wid_deg_mat)
        wid_deg_mat_inv[np.isinf(wid_deg_mat_inv)] = 0.

        # Random Walk normalized Laplacian
        # For SpectraConv
        # To [0, 1]
        # L_rw = D^{-1} * L_com = I - D^{-1} * A
        rw_normd_lap_mat = id_mat - np.matmul(deg_mat_inv, adj_mat)

        # For ChebConv
        # From [0, 1] to [-1, 1]
        # wid_L_rw = 2 * L_rw / lambda_max_rw - I
        #rw_max_lambda = max(np.linalg.eigvalsh(rw_normd_lap_mat))
        rw_max_lambda = max(eigvalsh(rw_normd_lap_mat))
        wid_rw_normd_lap_mat = 2 * rw_normd_lap_mat / rw_max_lambda - id_mat

        # For GCNConv
        # hat_L_rw = wid_D^{-1} * wid_A
        hat_rw_normd_lap_mat = np.matmul(wid_deg_mat_inv, wid_adj_mat)

        if mat_type == 'rw_normd_lap_mat':
            return rw_normd_lap_mat
        elif mat_type == 'wid_rw_normd_lap_mat':
            return wid_rw_normd_lap_mat
        elif mat_type == 'hat_rw_normd_lap_mat':
            return hat_rw_normd_lap_mat


def load_graphdata_channel1(graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, device, batch_size):
    """
    Load and preprocess the dataset for DSTAGNN.
    Args:
        graph_signal_matrix_filename: Path to the .npz file containing the dataset.
        num_of_hours: Number of past timesteps to use for prediction.
        num_of_days: Number of past days to use for prediction.
        num_of_weeks: Number of past weeks to use for prediction.
        device: Device to load the data onto (e.g., 'cuda:0' or 'cpu').
        batch_size: Batch size for the data loader.
    Returns:
        Data loaders and tensors for training, validation, and testing.
    """
    # Load the dataset
    data = np.load(graph_signal_matrix_filename)
    input_features = data['data']  # Shape: (num_timesteps, num_nodes, num_features)
    target = data['target']  # Shape: (num_timesteps, num_nodes)

    # Ensure the data is in the correct shape
    if len(input_features.shape) != 3:
        raise ValueError("Input data must have shape (num_timesteps, num_nodes, num_features)")
    if len(target.shape) != 2:
        raise ValueError("Target data must have shape (num_timesteps, num_nodes)")

    # Split the data into training, validation, and test sets
    num_samples = input_features.shape[0]
    train_size = int(num_samples * 0.6)
    val_size = int(num_samples * 0.2)

    train_data = input_features[:train_size]  # Shape: (train_size, num_nodes, num_features)
    val_data = input_features[train_size:train_size + val_size]  # Shape: (val_size, num_nodes, num_features)
    test_data = input_features[train_size + val_size:]  # Shape: (test_size, num_nodes, num_features)

    train_target = target[:train_size]  # Shape: (train_size, num_nodes)
    val_target = target[train_size:train_size + val_size]  # Shape: (val_size, num_nodes)
    test_target = target[train_size + val_size:]  # Shape: (test_size, num_nodes)

    # Normalize the data
    mean = train_data.mean(axis=(0, 1), keepdims=True)  # Shape: (1, 1, num_features)
    std = train_data.std(axis=(0, 1), keepdims=True)  # Shape: (1, 1, num_features)

    def normalize(x):
        return (x - mean) / std

    train_data = normalize(train_data)
    val_data = normalize(val_data)
    test_data = normalize(test_data)

    # Reshape the data to 4D: (num_samples, num_nodes, num_features, num_timesteps)
    # For train_data, we need to create sliding windows of size `num_of_hours`
    def create_windows(data, window_size):
        num_samples = data.shape[0] - window_size + 1
        windows = np.zeros((num_samples, data.shape[1], data.shape[2], window_size))
        for i in range(num_samples):
            windows[i] = data[i:i + window_size].transpose(1, 2, 0)  # Shape: (num_nodes, num_features, window_size)
        return windows

    train_data = create_windows(train_data, num_of_hours)  # Shape: (num_samples, num_nodes, num_features, num_of_hours)
    val_data = create_windows(val_data, num_of_hours)  # Shape: (num_samples, num_nodes, num_features, num_of_hours)
    test_data = create_windows(test_data, num_of_hours)  # Shape: (num_samples, num_nodes, num_features, num_of_hours)

    # For targets, we need to align them with the windows
    train_target = train_target[num_of_hours - 1:]  # Shape: (num_samples, num_nodes)
    val_target = val_target[num_of_hours - 1:]  # Shape: (num_samples, num_nodes)
    test_target = test_target[num_of_hours - 1:]  # Shape: (num_samples, num_nodes)

    # Convert to PyTorch tensors
    train_data = torch.FloatTensor(train_data).to(device)
    val_data = torch.FloatTensor(val_data).to(device)
    test_data = torch.FloatTensor(test_data).to(device)

    train_target = torch.FloatTensor(train_target).to(device)
    val_target = torch.FloatTensor(val_target).to(device)
    test_target = torch.FloatTensor(test_target).to(device)

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(train_data, train_target)
    val_dataset = torch.utils.data.TensorDataset(val_data, val_target)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_target)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_data, train_loader, train_target, val_data, val_loader, val_target, test_data, test_loader, test_target, mean, std

def compute_val_loss_mstgcn(net, val_loader, criterion, sw, epoch, limit=None):
    '''
    for rnn, compute mean loss on validation set
    :param net: model
    :param val_loader: torch.utils.data.utils.DataLoader
    :param criterion: torch.nn.MSELoss
    :param sw: tensorboardX.SummaryWriter
    :param global_step: int, current global_step
    :param limit: int,
    :return: val_loss
    '''

    net.train(False)  # ensure dropout layers are in evaluation mode

    with torch.no_grad():

        val_loader_length = len(val_loader)  # nb of batch

        tmp = []  # 记录了所有batch的loss

        for batch_index, batch_data in enumerate(val_loader):
            encoder_inputs, labels = batch_data
            outputs = net(encoder_inputs)
            loss = criterion(outputs, labels)  # 计算误差
            tmp.append(loss.item())
            if batch_index % 100 == 0:
                print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_loader_length, loss.item()))
            if (limit is not None) and batch_index >= limit:
                break

        validation_loss = sum(tmp) / len(tmp)
        sw.add_scalar('validation_loss', validation_loss, epoch)
    return validation_loss


def evaluate_on_test_mstgcn(net, test_loader, test_target_tensor, sw, epoch, _mean, _std):
    '''
    for rnn, compute MAE, RMSE, MAPE scores of the prediction for every time step on testing set.

    :param net: model
    :param test_loader: torch.utils.data.utils.DataLoader
    :param test_target_tensor: torch.tensor (B, N_nodes, T_output, out_feature)=(B, N_nodes, T_output, 1)
    :param sw:
    :param epoch: int, current epoch
    :param _mean: (1, 1, 3(features), 1)
    :param _std: (1, 1, 3(features), 1)
    '''

    net.train(False)  # ensure dropout layers are in test mode

    with torch.no_grad():

        test_loader_length = len(test_loader)

        test_target_tensor = test_target_tensor.cpu().numpy()

        prediction = []  # 存储所有batch的output

        for batch_index, batch_data in enumerate(test_loader):

            encoder_inputs, labels = batch_data

            outputs = net(encoder_inputs)

            prediction.append(outputs.detach().cpu().numpy())

            if batch_index % 100 == 0:
                print('predicting testing set batch %s / %s' % (batch_index + 1, test_loader_length))

        prediction = np.concatenate(prediction, 0)  # (batch, T', 1)
        prediction_length = prediction.shape[2]

        for i in range(prediction_length):
            assert test_target_tensor.shape[0] == prediction.shape[0]
            print('current epoch: %s, predict %s points' % (epoch, i))
            mae = mean_absolute_error(test_target_tensor[:, :, i], prediction[:, :, i])
            rmse = mean_squared_error(test_target_tensor[:, :, i], prediction[:, :, i]) ** 0.5
            mape = masked_mape_np(test_target_tensor[:, :, i], prediction[:, :, i], 0)
            print('MAE: %.2f' % (mae))
            print('RMSE: %.2f' % (rmse))
            print('MAPE: %.2f' % (mape))
            print()
            if sw:
                sw.add_scalar('MAE_%s_points' % (i), mae, epoch)
                sw.add_scalar('RMSE_%s_points' % (i), rmse, epoch)
                sw.add_scalar('MAPE_%s_points' % (i), mape, epoch)


def predict_and_save_results_mstgcn(net, data_loader, data_target_tensor, global_step, _mean, _std, params_path, type):
    '''

    :param net: nn.Module
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param epoch: int
    :param _mean: (1, 1, 3, 1)
    :param _std: (1, 1, 3, 1)
    :param params_path: the path for saving the results
    :return:
    '''
    net.train(False)  # ensure dropout layers are in test mode

    with torch.no_grad():

        data_target_tensor = data_target_tensor.cpu().numpy()

        loader_length = len(data_loader)  # nb of batch

        prediction = []  # 存储所有batch的output

        input = []  # 存储所有batch的input

        for batch_index, batch_data in enumerate(data_loader):

            encoder_inputs, labels = batch_data

            input.append(encoder_inputs[:, :, 0:1].cpu().numpy())  # (batch, T', 1)

            outputs = net(encoder_inputs)

            prediction.append(outputs.detach().cpu().numpy())

            if batch_index % 100 == 0:
                print('predicting data set batch %s / %s' % (batch_index + 1, loader_length))

        input = np.concatenate(input, 0)

        input = re_normalization(input, _mean, _std)

        prediction = np.concatenate(prediction, 0)  # (batch, T', 1)

        print('input:', input.shape)
        print('prediction:', prediction.shape)
        print('data_target_tensor:', data_target_tensor.shape)
        output_filename = os.path.join(params_path, 'output_epoch_%s_%s' % (global_step, type))
        np.savez(output_filename, input=input, prediction=prediction, data_target_tensor=data_target_tensor)

        # 计算误差
        excel_list = []
        prediction_length = prediction.shape[2]

        for i in range(prediction_length):
            assert data_target_tensor.shape[0] == prediction.shape[0]
            print('current epoch: %s, predict %s-th point' % (global_step, i+1))
            mae = mean_absolute_error(data_target_tensor[:, :, i], prediction[:, :, i])
            rmse = mean_squared_error(data_target_tensor[:, :, i], prediction[:, :, i]) ** 0.5
            mape = masked_mape_np(data_target_tensor[:, :, i], prediction[:, :, i], 0)
            print('MAE: %.2f' % (mae))
            print('RMSE: %.2f' % (rmse))
            print('MAPE: %.2f' % (mape))
            excel_list.extend([mae, rmse, mape])

        # print overall results
        mae = mean_absolute_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1))
        rmse = mean_squared_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
        mape = masked_mape_np(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
        print('all MAE: %.2f' % (mae))
        print('all RMSE: %.2f' % (rmse))
        print('all MAPE: %.2f' % (mape))
        excel_list.extend([mae, rmse, mape])
        print(excel_list)





