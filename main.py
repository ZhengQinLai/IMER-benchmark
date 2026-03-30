import json
import argparse
import csv
import os
from dotenv import load_dotenv
from trainer import train
import yaml
import numpy as np
import torch
import random

# Load environment variables from .env file
load_dotenv()

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    fix_seed(args['seed'])
    curve = train(args)  # curve is a list over sessions (length depends on enabled datasets)
    return curve, args['config'], args['split_mode']

def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def merge_config_and_args(config, args):
    config.update({k: v for k, v in vars(args).items() if v is not None})
    return config

def parse_args():
    parser = argparse.ArgumentParser(description='Reproduce of multiple pre-trained incremental learning algorithms.')
    parser.add_argument('--config', type=str, default='./exps/der.yaml',
                        help='YAML file of settings.')
    parser.add_argument('--log_file', type=str, default='_',
                        help='Log name.')
    parser.add_argument('--split_mode', type=str, default='SLCV',
                        help='split_mode, SLCV or ILCV')
    parser.add_argument('--device', type=str, default='1',
                        help='device')
    return parser.parse_args()  

def save_to_csv(curve, name, filename='result'):
    filename+='.csv'
    # Ensure curve is a numpy array
    curve = np.array(curve)
    if curve.ndim != 2:
        raise ValueError(f"Expected `curve` to be 2D (folds x sessions), got shape: {curve.shape}")
    
    # Compute row means and append as the fifth column
    row_means = curve.mean(axis=1)
    curve_with_row_means = np.column_stack((curve, row_means))
    
    # Compute column means (including the row means column) and append as the fifth row
    col_means = curve_with_row_means.mean(axis=0)
    result = np.vstack((curve_with_row_means, col_means))
    
    # Repeat `name` for each fold row; add an extra row for the overall mean.
    names_column = [name] * curve.shape[0] + ["Mean"]
    
    # Combine names and result
    final_result = np.column_stack((names_column, result))
    
    # Save to CSV
    with open(filename, mode='a+', newline='') as file:
        writer = csv.writer(file)
        header = ["Name"] + [f"session {i+1}" for i in range(curve.shape[1])] + ["Mean"]
        writer.writerow(header)
        writer.writerows(final_result)

if __name__ == '__main__':
    args = parse_args()  
    config = load_config(args.config)  
    final_config = merge_config_and_args(config, args)  

    curve, name, splitmode = main(final_config)
    
    # Save curve and name to result.csv
    save_to_csv(curve, name, splitmode+'tune')
