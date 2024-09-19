import sys

import torch
from torch.utils.data import DataLoader

sys.path.append('../')

from data_process.dataset import PIDA_dataset
from data_process.data_utils import get_all_detection_infor, train_val_data_separation
from data_process.transformation import Compose, Random_Crop, Resize, Normalize, Random_Rotate, Random_HFlip


def create_train_val_dataloader(configs):
    """Create dataloader for training and validate"""

    train_transform = Compose([
        Random_Crop(max_reduction_percent=0.15, p=0.5),
        Random_HFlip(p=0.5),
        Random_Rotate(rotation_angle_limit=10, p=0.5),
    ], p=1.)

    train_events_infor, val_events_infor, train_events_label, val_events_label = train_val_data_separation(configs)
    train_dataset = PIDA_dataset(train_events_infor, train_events_label, transform=train_transform,
                                  num_samples=configs.num_samples)
    train_sampler = None
    if configs.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=(train_sampler is None),
                                  pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=train_sampler)

    val_dataloader = None
    if not configs.no_val:
 
        val_transform = None
        val_sampler = None
        val_dataset = PIDA_dataset(val_events_infor, val_events_label, transform=val_transform,
                                    num_samples=configs.num_samples)
        if configs.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False,
                                    pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=val_sampler)

    return train_dataloader, val_dataloader, train_sampler


def create_test_dataloader(configs):
    """Create dataloader for testing phase"""

    test_transform = None
    dataset_type = 'test'
    test_events_infor, test_events_labels = get_all_detection_infor(configs.test_game_list, configs, dataset_type)
    test_dataset = PIDA_dataset(test_events_infor, test_events_labels, transform=test_transform,
                                 num_samples=configs.num_samples)
    test_sampler = None
    if configs.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False,
                                 pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=test_sampler)

    return test_dataloader


if __name__ == '__main__':
    from config.config import parse_configs

    configs = parse_configs()
    configs.distributed = False  # For testing
    train_dataloader, val_dataloader, train_sampler = create_train_val_dataloader(configs)
    print('len train_dataloader: {}, val_dataloader: {}'.format(len(train_dataloader), len(val_dataloader)))
    test_dataloader = create_test_dataloader(configs)
    print(f"len test_loader {len(test_dataloader)}")
