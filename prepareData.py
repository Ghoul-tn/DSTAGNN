import os
import numpy as np
import argparse
import configparser


def read_and_generate_dataset(graph_signal_matrix_filename,
                              num_of_weeks, num_of_days, num_of_hours,
                              num_for_predict, points_per_hour=12, save=False):
    '''
    Parameters
    ----------
    graph_signal_matrix_filename: str, path of graph signal matrix file
    num_of_weeks, num_of_days, num_of_hours: int
    num_for_predict: int
    points_per_hour: int, default 12, depends on data

    Returns
    ----------
    feature: np.ndarray,
             shape is (num_of_samples, num_of_depend * points_per_hour,
                       num_of_vertices, num_of_features)
    target: np.ndarray,
            shape is (num_of_samples, num_of_vertices, num_for_predict)
    '''
    # Load the input features and target
    data = np.load(graph_signal_matrix_filename)
    input_features = data['data']  # Shape: (num_timesteps, num_nodes, num_features)
    target = data['target']  # Shape: (num_timesteps, num_nodes)

    # Ensure the data is in the correct shape
    if len(input_features.shape) != 3:
        raise ValueError("Input data must have shape (num_timesteps, num_nodes, num_features)")
    if len(target.shape) != 2:
        raise ValueError("Target data must have shape (num_timesteps, num_nodes)")

    # Add a new dimension for num_features
    input_features = np.expand_dims(input_features, axis=-1)  # Shape: (num_timesteps, num_nodes, num_features, 1)

    # Split the data into training, validation, and test sets
    num_samples = input_features.shape[0]
    train_size = int(num_samples * 0.6)
    val_size = int(num_samples * 0.2)

    train_data = input_features[:train_size]
    val_data = input_features[train_size:train_size + val_size]
    test_data = input_features[train_size + val_size:]

    train_target = target[:train_size]
    val_target = target[train_size:train_size + val_size]
    test_target = target[train_size + val_size:]

    # Save the datasets (optional)
    if save:
        file = os.path.basename(graph_signal_matrix_filename).split('.')[0]
        dirpath = os.path.dirname("/kaggle/working/")
        filename = os.path.join(dirpath, f"{file}_r{num_of_hours}_d{num_of_days}_w{num_of_weeks}_dstagnn")
        print('save file:', filename)
        np.savez_compressed(filename,
                            train_x=train_data, train_target=train_target,
                            val_x=val_data, val_target=val_target,
                            test_x=test_data, test_target=test_target)

    return train_data, val_data, test_data, train_target, val_target, test_target


# Prepare dataset
parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/Gambia.conf', type=str,
                    help="configuration file path")
args = parser.parse_args()
config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
config.read(args.config)
data_config = config['Data']
training_config = config['Training']

adj_filename = data_config['adj_filename']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
if config.has_option('Data', 'id_filename'):
    id_filename = data_config['id_filename']
else:
    id_filename = None

num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
len_input = int(data_config['len_input'])
dataset_name = data_config['dataset_name']
num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])
num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']

# Load and preprocess the data
all_data = read_and_generate_dataset(graph_signal_matrix_filename, num_of_weeks, num_of_days, num_of_hours, num_for_predict, points_per_hour=points_per_hour, save=True)
