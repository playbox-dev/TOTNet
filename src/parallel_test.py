import torch
import random
import numpy as np
import os
import warnings
import torch.distributed as dist


from model.model_utils import make_data_parallel, get_num_parameters
from losses_metrics.losses import Heatmap_Ball_Detection_Loss, focal_loss, Heatmap_Ball_Detection_Loss_Weighted
from losses_metrics.metrics import heatmap_calculate_metrics, precision_recall_f1_tracknet, extract_coords, classification_metrics
from config.config import parse_configs
from utils.logger import Logger
from utils.train_utils import create_optimizer, create_lr_scheduler, get_saved_state, save_checkpoint, reduce_tensor, to_python_float, print_nvidia_driver_version
from utils.misc import AverageMeter, ProgressMeter, print_gpu_memory_usage
from data_process.dataloader import  create_occlusion_train_val_dataloader, create_occlusion_test_dataloader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

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
        torch.cuda.manual_seed_all(configs.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    if configs.gpu_idx is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if configs.dist_url == "env://" and configs.world_size == -1:
        configs.world_size = int(os.environ["WORLD_SIZE"])

    configs.distributed = configs.world_size > 1 or configs.multiprocessing_distributed

    configs.rank = int(os.environ["RANK"])
    configs.world_size = int(os.environ["WORLD_SIZE"])
    # Set the GPU for this process
    configs.gpu_idx = int(os.environ["LOCAL_RANK"])  # Rank within the current node
    configs.device = torch.device(f'cuda:{configs.gpu_idx}')
    torch.cuda.set_device(configs.device)

    if configs.distributed:
        dist.init_process_group(
            backend=configs.dist_backend,
            init_method=configs.dist_url,
            world_size=configs.world_size,
            rank=configs.rank
        )
    print(f"Rank {rank}/{configs.world_size} is using GPU {configs.device}.")
        

    main_worker(configs)



def main_worker(configs):

    print(f"Running on rank {configs.rank}, using GPU {configs.gpu_idx}")

    if configs.gpu_idx is not None:
        print("Use GPU: {} for training".format(configs.gpu_idx))
        configs.device = torch.device('cuda:{}'.format(configs.gpu_idx))

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
    
    model = torch.nn.Linear(10, 10).to(configs.device)
    
    torch.distributed.barrier()
    print(f"Rank {configs.rank}: Model built with {sum(p.numel() for p in model.parameters())} parameters.")

    try:
        # model = make_data_parallel(model, configs)
        ddp_model = DDP(model, device_ids=[configs.gpu_idx])
        print("Model made parallel successfully.")
    except RuntimeError as e:
        print(f"Runtime error during parallelization: {e}")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise


if __name__ == '__main__':
    main()
    print("complete building detector")


