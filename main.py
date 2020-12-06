import argparse
import datetime
import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

from model import Net
from train import train

parser = argparse.ArgumentParser()

# Architecture Flags
parser.add_argument('--input_dim', type=str, default='784,300,150')
parser.add_argument('--stripe_dim', type=str, default='300,5,3')
parser.add_argument('--num_stripes', type=str, default='1,30,12')
parser.add_argument('--num_active_stripes', type=str, default='1,3,2')
parser.add_argument('--distort_prob', type=str,
                    default='0.,.4,.4')  # Probability of stripe sparsity mask bits randomly flipping.
parser.add_argument('--distort_prob_decay', type=str, default='0,.025,.025')  # Lowers distort_prob by this amount every epoch.
parser.add_argument('--relax_stripe_sparsity', type=str,
                    default='0,2,1')  # Trains in final epoch with additional active stripes.

# Training Flags
parser.add_argument('--lr', type=float, default=.01)
parser.add_argument('--momentum', type=float, default=.9)
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--data_path', type=str, default='data.csv')
parser.add_argument('--log_path', type=str, default='logs')

args = vars(parser.parse_args())


def transform_data(net, data):
    data_size = len(data)
    with torch.no_grad():
        encoding = net.encode(torch.FloatTensor(data))
    return encoding.reshape(data_size, -1).detach().numpy()


def process_layer(criterion,
                  root_path,
                  X_train,
                  X_test,
                  Y_test,
                  num_epochs,
                  batch_size,
                  batch_no,
                  device,
                  input_dim,
                  stripe_dim,
                  num_stripes,
                  num_active_stripes,
                  distort_prob,
                  distort_prob_decay,
                  relax_stripe_sparsity):
    net = Net(input_dim,
              stripe_dim,
              num_stripes,
              num_active_stripes,
              distort_prob,
              device)
    optimizer = optim.SGD(net.parameters(),
                          lr=args['lr'],
                          momentum=args['momentum'])
    train(net,
          criterion,
          optimizer,
          root_path,
          X_train,
          X_test,
          Y_test,
          num_stripes,
          num_epochs,
          batch_size,
          batch_no,
          distort_prob_decay,
          relax_stripe_sparsity)
    return transform_data(net, X_train), transform_data(net, X_test)


def main(args):
    data = pd.read_csv(args['data_path']).values
    Y = data[:, :1].transpose()[0]
    X = data[:, 1:] / 255
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
    num_epochs = args['num_epochs']
    batch_size = args['batch_size']
    batch_no = len(X_train) // batch_size

    criterion = nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() and args['use_cuda_if_available'] else 'cpu')
    timestamp = str(datetime.datetime.now()).replace(' ', '_')
    root_path = os.path.join(args['log_path'], timestamp)
    print(f'Logging results to path:  {root_path}')

    int_param_list = ['input_dim', 'stripe_dim', 'num_stripes', 'num_active_stripes', 'relax_stripe_sparsity']
    int_params = [[int(num) for num in args[param].split(',')]
                  for param in int_param_list]
    float_param_list = ['distort_prob', 'distort_prob_decay']
    float_params = [[float(num) for num in args[param].split(',')]
                    for param in float_param_list]
    params = int_params + float_params
    
    layer_count = 0
    for (input_dim, stripe_dim, num_stripes, num_active_stripes, relax_stripe_sparsity,
         distort_prob, distort_prob_decay) in zip(*params):

        log_path = os.path.join(root_path, f'layer_{layer_count}')
        X_train, X_test = process_layer(criterion,
                                        root_path,
                                        X_train,
                                        X_test,
                                        Y_test,
                                        num_epochs,
                                        batch_size,
                                        batch_no,
                                        device,
                                        input_dim,
                                        stripe_dim,
                                        num_stripes,
                                        num_active_stripes,
                                        distort_prob,
                                        distort_prob_decay,
                                        relax_stripe_sparsity)
        layer_count += 1


if __name__ == '__main__':
    main(args)
