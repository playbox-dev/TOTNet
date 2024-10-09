import sys
import torch
import random
import numpy as np
import os
import warnings
import time
import torch.multiprocessing as mp
import torch.distributed as dist

from tqdm import tqdm
from model.deformable_detection_model import build_detector
from model.model_utils import make_data_parallel, get_num_parameters, post_process
from losses_metrics.losses import Heatmap_Ball_Detection_Loss
from losses_metrics.metrics import heatmap_calculate_metrics
from config.config import parse_configs
from utils.logger import Logger
from utils.train_utils import create_optimizer, create_lr_scheduler, get_saved_state, save_checkpoint, reduce_tensor, to_python_float
from utils.misc import AverageMeter, ProgressMeter, print_gpu_memory_usage
from data_process.dataloader import create_masked_train_val_dataloader, create_train_val_dataloader, create_masked_test_dataloader, create_test_dataloader, create_normal_train_val_dataloader
from torch.utils.tensorboard import SummaryWriter


def main():
    configs = parse_configs()

    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    device = torch.device(configs.device)

    # Re-produce results
    if configs.seed is not None:
        random.seed(configs.seed)
        np.random.seed(configs.seed)
        torch.manual_seed(configs.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
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
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            configs.rank = configs.rank * configs.ngpus_per_node + gpu_idx

        dist.init_process_group(backend=configs.dist_backend, init_method=configs.dist_url,
                                world_size=configs.world_size, rank=configs.rank)
        

    configs.is_master_node = (not configs.distributed) or (
            configs.distributed and (configs.rank % configs.ngpus_per_node == 0))

    if configs.is_master_node:
        logger = Logger(configs.logs_dir, configs.saved_fn)
        logger.info('>>> Created a new logger')
        logger.info('>>> configs: {}'.format(configs))
        tb_writer = SummaryWriter(log_dir=os.path.join(configs.logs_dir, 'tensorboard'))
    else:
        logger = None
        tb_writer = None
    

    model = build_detector(configs)

    model = make_data_parallel(model, configs)

    optimizer = create_optimizer(configs, model)
    lr_scheduler = create_lr_scheduler(optimizer, configs)
    best_val_loss = np.inf
    earlystop_count = 0
    is_best = False
    loss_func = Heatmap_Ball_Detection_Loss(h=configs.img_size[0], w=configs.img_size[1]).to(configs.device)


    if configs.is_master_node:
        num_parameters = get_num_parameters(model)
        logger.info('number of trained parameters of the model: {}'.format(num_parameters))

    if logger is not None:
        logger.info(">>> Loading dataset & getting dataloader...")
    # Create dataloader, option with normal data
    # train_loader, val_loader, train_sampler = create_masked_train_val_dataloader(configs)
    train_loader, val_loader, train_sampler = create_normal_train_val_dataloader(configs)
    test_loader = create_masked_test_dataloader(configs)

    # Print the number of samples for this GPU/worker
    if configs.distributed:
        print(f"GPU {configs.gpu_idx} (Rank {configs.rank}): {len(train_loader.dataset)} samples total, {len(train_loader.sampler)} samples for this GPU")
    
    if logger is not None:
        logger.info('number of batches in train set: {}'.format(len(train_loader)))
        if val_loader is not None:
            logger.info('number of batches in val set: {}'.format(len(val_loader)))
        logger.info('number of batches in test set: {}'.format(len(test_loader)))

    
    for epoch in range(configs.start_epoch, configs.num_epochs + 1):
        # Get the current learning rate
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        if logger is not None:
            logger.info('{}'.format('*-' * 40))
            logger.info('{} {}/{} {}'.format('=' * 35, epoch, configs.num_epochs, '=' * 35))
            logger.info('{}'.format('*-' * 40))
            logger.info('>>> Epoch: [{}/{}] learning rate: {:.2e}'.format(epoch, configs.num_epochs, lr))

        if configs.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
    
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_func, epoch, configs, logger)
        loss_dict = {'train': train_loss}

        if configs.no_val == False:
            val_loss = evaluate_one_epoch(val_loader, model, loss_func, epoch, configs, logger)
            is_best = val_loss <= best_val_loss
            best_val_loss = min(val_loss, best_val_loss)
            loss_dict['val'] = val_loss

        if configs.no_test == False:
            test_loss = evaluate_one_epoch(test_loader, model, loss_func, epoch, configs, logger)
            loss_dict['test'] = test_loss
        # Write tensorboard
        if tb_writer is not None:
            tb_writer.add_scalars('Loss', loss_dict, epoch)
        # Save checkpoint
        if configs.is_master_node and (is_best or ((epoch % configs.checkpoint_freq) == 0)):
            saved_state = get_saved_state(model, optimizer, lr_scheduler, epoch, configs, best_val_loss,
                                          earlystop_count)
            save_checkpoint(configs.checkpoints_dir, configs.saved_fn, saved_state, is_best, epoch)
        # Check early stop training
        if configs.earlystop_patience is not None:
            earlystop_count = 0 if is_best else (earlystop_count + 1)
            print_string = ' |||\t earlystop_count: {}'.format(earlystop_count)
            if configs.earlystop_patience <= earlystop_count:
                print_string += '\n\t--- Early stopping!!!'
                break
            else:
                print_string += '\n\t--- Continue training..., earlystop_count: {}'.format(earlystop_count)
            if logger is not None:
                logger.info(print_string)
        # Adjust learning rate
        if configs.lr_type == 'plateau':
            assert (not configs.no_val), "Only use plateau when having validation set"
            lr_scheduler.step(val_loss)
        else:
            lr_scheduler.step()

    if tb_writer is not None:
        tb_writer.close()
    if configs.distributed:
        cleanup()

def cleanup():
    dist.destroy_process_group()


def train_one_epoch(train_loader, model, optimizer, loss_func, epoch, configs, logger):
    configs = parse_configs()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses],
                             prefix="Train - Epoch: [{}/{}]".format(epoch, configs.num_epochs))

    # switch to train mode
    model.train()
    start_time = time.time()
    
    for batch_idx, (batch_data, (masked_frameids, _, labels)) in enumerate(tqdm(train_loader)):

        data_time.update(time.time() - start_time)

        batch_size = batch_data.size(0)
        batch_data = batch_data.to(configs.device)
        labels = labels.to(configs.device)
        labels = labels.float()

        output_coords = model(batch_data.float()) # output in shape ([B,W],[B,H]) if output heatmap
        total_loss = loss_func(output_coords, labels)

        # For torch.nn.DataParallel case
        if (not configs.distributed) and (configs.gpu_idx is None):
            total_loss = torch.mean(total_loss)

        # zero the parameter gradients
        optimizer.zero_grad()
        # compute gradient and perform backpropagation
        total_loss.backward()
        optimizer.step()

        if configs.distributed:
            reduced_loss = reduce_tensor(total_loss.data, configs.world_size)
        else:
            reduced_loss = total_loss.data
        losses.update(to_python_float(reduced_loss), batch_size)
        # measure elapsed time
        torch.cuda.synchronize()
        batch_time.update(time.time() - start_time)
        # Log message
        if logger is not None:
            if ((batch_idx + 1) % configs.print_freq) == 0:
                logger.info(progress.get_message(batch_idx))
        start_time = time.time()

    return losses.avg

def evaluate_one_epoch(val_loader, model, loss_func, epoch, configs, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    rmses = AverageMeter('RMSE', ':.4e')

    progress = ProgressMeter(len(val_loader), [batch_time, data_time, losses, rmses],
                             prefix="Evaluate - Epoch: [{}/{}]".format(epoch, configs.num_epochs))
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        for batch_idx, (batch_data, (masked_frameids, masked_frames, labels)) in enumerate(tqdm(val_loader)):
            data_time.update(time.time() - start_time)
            batch_size = batch_data.size(0)

            batch_data = batch_data.to(configs.device)
            labels = labels.to(configs.device)
            labels = labels.float()

            # calculate loss
            output_coords_logits = model(batch_data.float())
            total_loss = loss_func(output_coords_logits, labels)

            mse, rmse, mae, euclidean_distance = heatmap_calculate_metrics(output_coords_logits, labels, img_height=configs.img_size[0], img_width=configs.img_size[1])
            rmse_tensor = torch.tensor(rmse).to(configs.device)

            # For torch.nn.DataParallel case
            if (not configs.distributed) and (configs.gpu_idx is None):
                total_loss = torch.mean(total_loss)

            if configs.distributed:
                reduced_loss = reduce_tensor(total_loss.data, configs.world_size)
                reduced_rmse = reduce_tensor(rmse_tensor, configs.world_size)
            else:
                reduced_loss = total_loss.data
                reduced_rmse = rmse
            losses.update(to_python_float(reduced_loss), batch_size)
            rmses.update(to_python_float(reduced_rmse), batch_size)
            # measure elapsed time
            torch.cuda.synchronize()
            batch_time.update(time.time() - start_time)

            # Log message
            if logger is not None:
                if ((batch_idx + 1) % configs.print_freq) == 0:
                    logger.info(progress.get_message(batch_idx))

            start_time = time.time()

    return losses.avg

if __name__ == '__main__':
    main()
    print("complete building detector")


