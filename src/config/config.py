import torch
import argparse
import datetime
import os
import sys
from easydict import EasyDict as edict


def parse_configs():
    parser = argparse.ArgumentParser(description='PIDA')
    parser.add_argument('--seed', type=int, default=2024,
                    help='re-produce the results with seed random')
    parser.add_argument('--working-dir', type=str, default='../', metavar='PATH',
                        help='the ROOT working directory')
    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--smooth-labelling', action='store_true',
                    help='If true, smoothly make the labels of event spotting')
    parser.add_argument('--no-val', action='store_true',
                    help='If true, use all data for training, no validation set')
    parser.add_argument('--val-size', type=float, default=0.2,
                    help='The size of validation set')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 8), this is the total'
                            'batch size of all GPUs on the current node when using'
                            'Data Parallel or Distributed Data Parallel')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of threads for loading data')
    parser.add_argument('--distributed', type=bool, default=False,
                        help="if its trained using multiple gpu")

    configs = edict(vars(parser.parse_args()))
    ####################################################################
    ############## Hardware configurations ############################
    ####################################################################
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda')
    configs.ngpus_per_node = torch.cuda.device_count()

    configs.pin_memory = True

    configs.org_size = (1920, 1080)
    configs.num_frames_sequence = 9

    configs.results_dir = os.path.join(configs.working_dir, 'results')

    ####################################################################
    ##############     Data configs            ###################
    ####################################################################

    # configs.dataset_dir = os.path.join(configs.working_dir, 'dataset')
    configs.dataset_dir = os.path.join('/home/s224705071/github/TT/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch/', 'dataset')
    configs.train_game_list = ['game_1', 'game_2', 'game_3', 'game_4', 'game_5']
    configs.test_game_list = ['test_1', 'test_2', 'test_3', 'test_4', 'test_5', 'test_6', 'test_7']
    configs.events_dict = {
        'bounce': 0,
        'net': 1,
        'empty_event': 2
    }
    configs.events_weights_loss_dict = {
        'bounce': 1.,
        'net': 3.,
    }

    return configs


if __name__ == "__main__":
    configs = parse_configs()
    print(configs)

    print(datetime.date.today())
    print(datetime.datetime.now().year)
