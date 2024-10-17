import time
import sys
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
import math
from tqdm import tqdm

sys.path.append('./')

from data_process.dataloader import create_test_dataloader, create_normal_test_dataloader, create_masked_test_dataloader
from model.model_utils import make_data_parallel, get_num_parameters, load_pretrained_model
from model.deformable_detection_model import build_detector
from losses_metrics.metrics import heatmap_calculate_metrics, calculate_rmse
from utils.misc import AverageMeter
from config.config import parse_configs



def main():
    configs = parse_configs()

    if configs.gpu_idx is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if configs.dist_url == "env://" and configs.world_size == -1:
        configs.world_size = int(os.environ["WORLD_SIZE"])

    configs.distributed = configs.world_size > 1 or configs.multiprocessing_distributed

    if configs.multiprocessing_distributed:
        configs.world_size = configs.ngpus_per_node * configs.world_size
        mp.spawn(main_worker, nprocs=configs.ngpus_per_node, args=(configs,))
    else:
        main_worker(configs.gpu_idx, configs)


def main_worker(gpu_idx, configs):
    configs.gpu_idx = gpu_idx

    if configs.gpu_idx is not None:
        print("Use GPU: {} for training".format(configs.gpu_idx))
        configs.device = torch.device('cuda:{}'.format(configs.gpu_idx))

    if configs.distributed:
        if configs.dist_url == "env://" and configs.rank == -1:
            configs.rank = int(os.environ["RANK"])
        if configs.multiprocessing_distributed:
            configs.rank = configs.rank * configs.ngpus_per_node + gpu_idx

        dist.init_process_group(backend=configs.dist_backend, init_method=configs.dist_url,
                                world_size=configs.world_size, rank=configs.rank)

    configs.is_master_node = (not configs.distributed) or (
            configs.distributed and (configs.rank % configs.ngpus_per_node == 0))

    model = build_detector(configs)

    model = make_data_parallel(model, configs)

    if configs.is_master_node:
        num_parameters = get_num_parameters(model)
        print('number of trained parameters of the model: {}'.format(num_parameters))

    if configs.pretrained_path is not None:
        model = load_pretrained_model(model, configs.pretrained_path, gpu_idx)
    # Load dataset
    # test_loader = create_normal_test_dataloader(configs)
    test_loader = create_masked_test_dataloader(configs)
    test(test_loader, model, configs)


def test(test_loader, model, configs):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    rmse_overall = AverageMeter('RMSE_Overall', ':6.4f')
    real_rmse = AverageMeter('Real_RMSE', ':6.4f')

    x_scale = configs.org_size[1]/configs.img_size[1]
    y_scale = configs.org_size[0]/configs.img_size[0]

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        for batch_idx, (batch_data, (masked_frameids, masked_frames, labels)) in enumerate(tqdm(test_loader)):

            print(f'\n===================== batch_idx: {batch_idx} ================================')

            data_time.update(time.time() - start_time)
            batch_size = batch_data.size(0)

            batch_data = batch_data.to(configs.device)
            labels = labels.to(configs.device)
            labels = labels.float()
            # compute output

            output_coords_logits = model(batch_data.float())

            mse, rmse, mae, euclidean_distance = heatmap_calculate_metrics(output_coords_logits, labels, img_height=configs.img_size[0], img_width=configs.img_size[1])
            pred_x_logits, pred_y_logits = output_coords_logits

            for sample_idx in range(batch_size):
                pred_x_logit = pred_x_logits[sample_idx]  # Shape: [W]
                pred_y_logit = pred_y_logits[sample_idx]  # Shape: [H]
                label = labels[sample_idx]
                # Predicted coordinates are extracted by taking the argmax over logits
                x_pred_indice = torch.argmax(pred_x_logit, dim=0)  # [W] -> scalar representing the predicted x index
                y_pred_indice = torch.argmax(pred_y_logit, dim=0)  # [H] -> scalar representing the predicted y index
                # Convert indices to float for calculations
                x_pred = x_pred_indice.float()
                y_pred = y_pred_indice.float()

                original_x, original_y = label[0].item()*x_scale, label[1].item()*y_scale
                rescaled_x_pred, rescaled_y_pred = x_pred*x_scale, y_pred*y_scale
                original_rmse = calculate_rmse(original_x, original_y, rescaled_x_pred, rescaled_y_pred)
                real_rmse.update(original_rmse)

                print('Ball Detection - \t Overall: \t (x, y) - org: ({}, {}), prediction = ({}, {}), Original Size org({},{}), prediction = ({},{})'.format(
                        label[0].item(), label[1].item(), x_pred.item(), y_pred.item(), original_x, original_y, rescaled_x_pred, rescaled_y_pred))

                if ((batch_idx + 1) % configs.print_freq) == 0:
                    print(
                        'batch_idx: {}  rmse_global: {:.1f}, real_rmse {:.1f}'.format(batch_idx, rmse, original_rmse))

                batch_time.update(time.time() - start_time)
                start_time = time.time()
            rmse_overall.update(rmse)

    print(
        'rmse_global: {:.1f}, real_rmse {:.1f}'.format(rmse_overall.avg, real_rmse.avg))
    print('Done testing')


if __name__ == '__main__':
    main()
