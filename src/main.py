import torch
import random
import numpy as np
import os
import warnings
import time
import torch.distributed as dist

from collections import Counter
from tqdm import tqdm
# from model.deformable_detection_model import build_detector
# from model.propose_model import build_detector
from model.tracknet import build_TrackerNet, build_TrackNetV2
from model.wasb import build_wasb
from model.motion_model import build_motion_model, post_process
from model.motion_model_light import build_motion_model_light
from model.motion_model_v3 import build_motion_model_light_opticalflow
from model.mamba_model import build_mamba
from model.two_stream_network import build_two_streams_model
from model.model_utils import make_data_parallel, get_num_parameters
from losses_metrics.losses import Heatmap_Ball_Detection_Loss, focal_loss
from losses_metrics.metrics import heatmap_calculate_metrics, precision_recall_f1_tracknet, extract_coords, classification_metrics
from losses_metrics.physics_loss import PhysicsLoss
from config.config import parse_configs
from utils.logger import Logger
from utils.train_utils import create_optimizer, create_lr_scheduler, get_saved_state, save_checkpoint, reduce_tensor, to_python_float, print_nvidia_driver_version
from utils.misc import AverageMeter, ProgressMeter, print_gpu_memory_usage
from data_process.dataloader import  create_occlusion_train_val_dataloader, create_occlusion_test_dataloader
from torch.utils.tensorboard import SummaryWriter


# torch.autograd.set_detect_anomaly(True)

def main():
    configs = parse_configs()

    rank = int(os.environ.get("RANK", 0))  # Default to 0 if RANK is not set

    if torch.cuda.is_available() and rank==0:
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

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

    main_worker(configs)



def main_worker(configs):

    configs.rank = int(os.environ["RANK"])
    configs.world_size = int(os.environ["WORLD_SIZE"])
    # Set the GPU for this process
    configs.gpu_idx = configs.rank % torch.cuda.device_count()  # Map rank to available GPU
    configs.device = torch.device(f'cuda:{configs.gpu_idx}')

    print(f"Running on rank {configs.rank}, using GPU {configs.gpu_idx}")

    if configs.gpu_idx is not None:
        print("Use GPU: {} for training".format(configs.gpu_idx))
        configs.device = torch.device('cuda:{}'.format(configs.gpu_idx))


    if configs.distributed:
        dist.init_process_group(
            backend=configs.dist_backend,
            init_method=configs.dist_url,
            world_size=configs.world_size,
            rank=configs.rank
        )
        

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
    
    if configs.model_choice == 'wasb':
        print("Building WASB model...")
        model = build_wasb(configs)
    elif configs.model_choice == 'tracknetv2':
        print("Building TrackNetV2 model...")
        model = build_TrackNetV2(configs)
    elif configs.model_choice == 'mamba':
        print("Building Mamba model...")
        model = build_mamba(configs)
    elif configs.model_choice == 'motion':
        print("Building Motion model...")
        model = build_motion_model(configs)
    elif configs.model_choice == 'two_stream_model':
        print("Building Two Streams model...")
        model = build_two_streams_model(configs)
    elif configs.model_choice == 'motion_light':
        print("Building Motion Light model...")
        model = build_motion_model_light(configs)
    elif configs.model_choice == 'motion_light_opticalflow':
        print("Building Motion Light Optical Flow model...")
        model = build_motion_model_light_opticalflow(configs)
    else:
        raise ValueError(f"Unknown model choice: {configs.model_choice}")

    model = make_data_parallel(model, configs)

    optimizer = create_optimizer(configs, model)
    lr_scheduler = create_lr_scheduler(optimizer, configs)
    scaler = torch.amp.GradScaler()
    best_val_loss = np.inf
    earlystop_count = 0
    is_best = False

    loss_func = Heatmap_Ball_Detection_Loss().to(configs.device)

    if configs.is_master_node:
        num_parameters = get_num_parameters(model)
        logger.info('number of trained parameters of the model: {}'.format(num_parameters))

    if logger is not None:
        logger.info(">>> Loading dataset & getting dataloader...")
    # Create dataloader, option with normal data
    
    train_loader, val_loader, train_sampler = create_occlusion_train_val_dataloader(configs, subset_size=configs.num_samples)

    # train_visibility_counts = Counter()
    # for _, (_, _,  visibility, _,) in train_loader:
    #     visibility =  visibility.view(-1).numpy() if torch.is_tensor( visibility) else  visibility
    #     train_visibility_counts.update( visibility)
    
    # val_visibility_counts = Counter()
    # for _, (_, _,  visibility, _,) in val_loader:
    #     visibility =  visibility.view(-1).numpy() if torch.is_tensor( visibility) else  visibility
    #     val_visibility_counts.update( visibility)

    # print(f"Train visibility counter {train_visibility_counts}, Val visibility counter {val_visibility_counts}")

    # test_loader = create_occlusion_test_dataloader(configs, configs.num_samples)
    # test_visibility_counts = Counter()
    # for _, (_, _,  visibility, _,) in test_loader:
    #     visibility =  visibility.view(-1).numpy() if torch.is_tensor( visibility) else  visibility
    #     test_visibility_counts.update( visibility)
    # print(f"test visibility counter {test_visibility_counts}")

    # exit()

    test_loader = None
    if configs.no_test == False:
        test_loader = create_occlusion_test_dataloader(configs, configs.num_samples)

    if logger is not None:
        batch_data, _ = next(iter(train_loader))
        logger.info(f"Batch data shape: {batch_data.shape}")

    # Print the number of samples for this GPU/worker
    if configs.distributed:
        print(f"GPU {configs.gpu_idx} (Rank {configs.rank}): {len(train_loader.dataset)} samples total, {len(train_loader.sampler)} samples for this GPU")
    
    if logger is not None:
        logger.info('number of batches in train set: {}'.format(len(train_loader)))
        if val_loader is not None:
            logger.info('number of batches in val set: {}'.format(len(val_loader)))
        if test_loader is not None:
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
    
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_func, scaler, epoch, configs, logger)
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
        if configs.is_master_node:
            saved_state = get_saved_state(model, optimizer, lr_scheduler, epoch, configs, best_val_loss,
                                        earlystop_count)
            
            # Save checkpoint if it's the best
            if is_best:
                save_checkpoint(configs.checkpoints_dir, configs.saved_fn, saved_state, is_best=True, epoch=epoch)
                logger.info(f"Best checkpoint has been saved at epoch {epoch}")
            
            # Save checkpoint based on checkpoint frequency
            if (epoch % configs.checkpoint_freq) == 0:
                save_checkpoint(configs.checkpoints_dir, configs.saved_fn, saved_state, is_best=False, epoch=epoch)
                logger.info(f"Checkpoint has been saved at epoch {epoch}")
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
            if configs.no_val:
                lr_scheduler.step(test_loss)
            else:
                lr_scheduler.step(val_loss)
        else:
            lr_scheduler.step()

    if tb_writer is not None:
        tb_writer.close()
    if configs.distributed:
        cleanup()

def cleanup():
    dist.destroy_process_group()


def train_one_epoch(train_loader, model, optimizer, loss_func, scaler, epoch, configs, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    heatmap_losses = AverageMeter('Heatmap Loss', ':.4e')
    cls_losses = AverageMeter('Classification Loss', ':.4e')

    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, heatmap_losses, cls_losses],
                             prefix="Train - Epoch: [{}/{}]".format(epoch, configs.num_epochs))

    # switch to train mode
    model.train()
    start_time = time.time()
    
    for batch_idx, (batch_data, (_, labels, visibiltity, status)) in enumerate(tqdm(train_loader)):
        data_time.update(time.time() - start_time)

        batch_size = batch_data.size(0)
        batch_data = batch_data.to(configs.device, dtype=torch.float)
        labels = labels.to(configs.device, dtype=torch.float)
        visibiltity = visibiltity.to(configs.device)

        if configs.model_choice == 'tracknet' or  configs.model_choice == 'tracknetv2' or configs.model_choice == 'wasb':
            # #for tracknet we need to rehsape the data
            B, N, C, H, W = batch_data.shape
            # Permute to bring frames and channels together
            batch_data = batch_data.permute(0, 2, 1, 3, 4).contiguous()  # Shape: [B, C, N, H, W]
            # Reshape to combine frames into the channel dimension
            batch_data = batch_data.view(B, N * C, H, W)  # Shape: [B, N*C, H, W]


        with torch.autocast(device_type='cuda'):
            output_heatmap, cls_score = model(batch_data) # output in shape ([B, W],[B, H]) if output heatmap
            if torch.isnan(output_heatmap[0]).any() or torch.isnan(output_heatmap[1]).any():
                if logger is not None:
                    logger.info("nan appeard here")
                output_heatmap = [
                    torch.nan_to_num(output_heatmap[0]),
                    torch.nan_to_num(output_heatmap[1]),
                ]
        if cls_score == None:
            cls_loss = torch.tensor(1e-8, device=configs.device)
        else:
            cls_loss = focal_loss(cls_score, visibiltity) * 0.1
        heatmap_loss = loss_func(output_heatmap, labels)
        total_loss = heatmap_loss + cls_loss


        # For torch.nn.DataParallel case
        if (not configs.distributed) and (configs.gpu_idx is None):
            total_loss = torch.mean(total_loss)

        # zero the parameter gradients
        optimizer.zero_grad()
        # compute gradient and perform backpropagation
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if configs.distributed:
            reduced_heatmap_loss = reduce_tensor(heatmap_loss, configs.world_size)
            reduced_loss = reduce_tensor(total_loss, configs.world_size)
            reduced_cls_loss = reduce_tensor(cls_loss, configs.world_size)
        else:
            reduced_heatmap_loss = heatmap_loss
            reduced_loss = total_loss
            reduced_cls_loss = cls_loss
        losses.update(to_python_float(reduced_loss), batch_size)
        heatmap_losses.update(to_python_float(reduced_heatmap_loss), batch_size)
        cls_losses.update(to_python_float(reduced_cls_loss), batch_size)
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
    heatmap_losses = AverageMeter('Heatmap Loss', ':.4e')
    cls_losses = AverageMeter('Classification Loss', ':.4e')

    rmses = AverageMeter('RMSE', ':.4e')
    accuracy_overall = AverageMeter('Accuracy', ':6.4f')
    precision_overall = AverageMeter('Precision', ':6.4f')
    recall_overall = AverageMeter('Recall', ':6.4f')
    f1_overall = AverageMeter('F1', ':6.4f')

    cls_accuracy_overall = AverageMeter('Cls Accuracy', ':6.4f')
    cls_precision_overall = AverageMeter('Cls Precision', ':6.4f')
    cls_recall_overall = AverageMeter('Cls Recall', ':6.4f')
    cls_f1_overall = AverageMeter('Cls F1', ':6.4f')


    progress = ProgressMeter(len(val_loader), [batch_time, data_time, losses, heatmap_losses, cls_losses, rmses, accuracy_overall, precision_overall, recall_overall, f1_overall,
                                               cls_accuracy_overall, cls_precision_overall, cls_recall_overall, cls_f1_overall],
                             prefix="Evaluate - Epoch: [{}/{}]".format(epoch, configs.num_epochs))
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        for batch_idx, (batch_data, (_, labels, visibility, status)) in enumerate(tqdm(val_loader)):
            data_time.update(time.time() - start_time)
            batch_size = batch_data.size(0)

            batch_data = batch_data.to(configs.device, dtype=torch.float)
            labels = labels.to(configs.device, dtype=torch.float)
            visibility = visibility.to(configs.device)

            if configs.model_choice == 'tracknet' or  configs.model_choice == 'tracknetv2' or configs.model_choice == 'wasb':
                # #for tracknet we need to rehsape the data
                B, N, C, H, W = batch_data.shape
                # Permute to bring frames and channels together
                batch_data = batch_data.permute(0, 2, 1, 3, 4).contiguous()  # Shape: [B, C, N, H, W]
                # Reshape to combine frames into the channel dimension
                batch_data = batch_data.view(B, N * C, H, W)  # Shape: [B, N*C, H, W]

            with torch.autocast(device_type='cuda'):
                output_heatmap, cls_score = model(batch_data) # output in shape ([B, W],[B, H]) if output heatmap, just raw logits
                if torch.isnan(output_heatmap[0]).any() or torch.isnan(output_heatmap[1]).any():
                    if logger is not None:
                        logger.info("nan appeard here")
                    output_heatmap = [
                        torch.nan_to_num(output_heatmap[0]),
                        torch.nan_to_num(output_heatmap[1]),
                    ]

            if cls_score == None:
                cls_loss = torch.tensor(1e-8, device=configs.device)
            else:
                cls_loss = focal_loss(cls_score, visibility) * 0.1 
            heatmap_loss = loss_func(output_heatmap, labels)
            total_loss = heatmap_loss + cls_loss

            mse, rmse, mae, euclidean_distance = heatmap_calculate_metrics(output_heatmap, labels)
            post_processed_coords = extract_coords(output_heatmap)
            precision, recall, f1, accuracy = precision_recall_f1_tracknet(post_processed_coords, labels, distance_threshold=configs.ball_size)
            if cls_score == None:
                cls_accuracy_tensor = torch.tensor(1e-8, device=configs.device)
                cls_precision_tensor = torch.tensor(1e-8, device=configs.device)
                cls_recall_tensor = torch.tensor(1e-8, device=configs.device)
                cls_f1_tensor = torch.tensor(1e-8, device=configs.device)
            else:
                cls_dict = classification_metrics(cls_score, visibility)
                cls_accuracy_tensor = torch.tensor(cls_dict['accuracy'], device=configs.device)
                cls_precision_tensor = torch.tensor(cls_dict['precision'], device=configs.device)
                cls_recall_tensor = torch.tensor(cls_dict['recall'], device=configs.device)
                cls_f1_tensor = torch.tensor(cls_dict['f1_score'], device=configs.device)

            rmse_tensor = torch.tensor(rmse).to(configs.device)
            precision_tensor = torch.tensor(precision).to(configs.device)
            recall_tensor = torch.tensor(recall).to(configs.device)
            f1_tensor = torch.tensor(f1).to(configs.device)
            accuracy_tensor = torch.tensor(accuracy).to(configs.device)
           

            # For torch.nn.DataParallel case
            if (not configs.distributed) and (configs.gpu_idx is None):
                total_loss = torch.mean(total_loss)

            if configs.distributed:
                reduced_loss = reduce_tensor(total_loss, configs.world_size)
                reduced_heatmap_loss = reduce_tensor(heatmap_loss, configs.world_size)
                reduced_cls_loss = reduce_tensor(cls_loss, configs.world_size)
                reduced_rmse = reduce_tensor(rmse_tensor, configs.world_size)
                reduced_accuracy = reduce_tensor(accuracy_tensor, configs.world_size)
                reduced_precision = reduce_tensor(precision_tensor, configs.world_size)
                reduced_recall = reduce_tensor(recall_tensor, configs.world_size)
                reduced_f1 = reduce_tensor(f1_tensor, configs.world_size)
                reduced_cls_accuracy = reduce_tensor(cls_accuracy_tensor, configs.world_size)
                reduced_cls_precision = reduce_tensor(cls_precision_tensor, configs.world_size)
                reduced_cls_recall = reduce_tensor(cls_recall_tensor, configs.world_size)
                reduced_cls_f1 = reduce_tensor(cls_f1_tensor, configs.world_size)
                
            else:
                reduced_heatmap_loss = heatmap_loss
                reduced_cls_loss = cls_loss
                reduced_rmse = rmse
                reduced_accuracy = accuracy
                reduced_precision = precision
                reduced_recall = recall
                reduced_f1 = f1
                reduced_loss = total_loss
                reduced_cls_accuracy = cls_accuracy_tensor
                reduced_cls_precision = cls_precision_tensor
                reduced_cls_recall = cls_recall_tensor
                reduced_cls_f1 = cls_f1_tensor

            losses.update(to_python_float(reduced_loss), batch_size)
            heatmap_losses.update(to_python_float(reduced_heatmap_loss), batch_size)
            cls_losses.update(to_python_float(reduced_cls_loss), batch_size)
            rmses.update(to_python_float(reduced_rmse), batch_size)
            accuracy_overall.update(to_python_float(reduced_accuracy), batch_size)
            precision_overall.update(to_python_float(reduced_precision), batch_size)
            recall_overall.update(to_python_float(reduced_recall), batch_size)
            f1_overall.update(to_python_float(reduced_f1), batch_size)

            cls_accuracy_overall.update(to_python_float(reduced_cls_accuracy), batch_size)
            cls_precision_overall.update(to_python_float(reduced_cls_precision), batch_size)
            cls_recall_overall.update(to_python_float(reduced_cls_recall), batch_size)
            cls_f1_overall.update(to_python_float(reduced_cls_f1), batch_size)

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


