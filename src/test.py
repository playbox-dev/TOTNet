import time
import sys
import os
import warnings
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from tqdm import tqdm

sys.path.append('./')

from data_process.dataloader import create_occlusion_test_dataloader, create_occlusion_train_val_dataloader
from model.model_utils import make_data_parallel, get_num_parameters, load_pretrained_model
# from model.deformable_detection_model import build_detector
# from model.propose_model import build_detector
from model.tracknet import build_TrackerNet, build_TrackNetV2
from model.mamba_model import build_mamba
from model.two_stream_network import build_two_streams_model
from model.wasb import build_wasb
from model.motion_model import build_motion_model
from losses_metrics.metrics import heatmap_calculate_metrics, calculate_rmse, precision_recall_f1_tracknet, extract_coords, classification_metrics
from utils.misc import AverageMeter
from utils.visualization import visualize_and_save_2d_heatmap
from config.config import parse_configs



def main():
    configs = parse_configs()
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

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

    if configs.model_choice == 'wasb':
        model = build_wasb(configs)
    if configs.model_choice == 'tracknetv2':
        model = build_TrackNetV2(configs)
    if configs.model_choice == 'mamba':
        model = build_mamba(configs)
    if configs.model_choice == 'motion':
        model = build_motion_model(configs)
    if configs.model_choice == 'two_stream_model':
        model = build_two_streams_model(configs)


    model = make_data_parallel(model, configs)

    if configs.is_master_node:
        num_parameters = get_num_parameters(model)
        print('number of trained parameters of the model: {}'.format(num_parameters))

    if configs.pretrained_path is not None:
        model = load_pretrained_model(model, configs.pretrained_path, gpu_idx)
    # Load dataset
    # test_loader = create_normal_test_dataloader(configs)
    # train_loader, val_loader, train_sampler = create_occlusion_train_val_dataloader(configs, subset_size=configs.num_samples)
    # print(f"number of batches in test is {len(val_loader)}")
    # test(val_loader, model, configs)

    test_loader = create_occlusion_test_dataloader(configs, configs.num_samples)
    print(f"number of batches in test is {len(test_loader)}")
    test(test_loader, model, configs)


def test(test_loader, model, configs):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    rmse_overall = AverageMeter('RMSE_Overall', ':6.4f')
    real_rmse = AverageMeter('Real_RMSE', ':6.4f')
    accuracy_overall = AverageMeter('Accuracy', '6.4f')
    percision_overall = AverageMeter('Percision', '6.4f')
    recall_overall = AverageMeter('Recall', '6.4f')
    f1_overall = AverageMeter('F1', '6.4f')
    cls_accuracy_overall = AverageMeter('Cls Accuracy', ':6.4f')
    cls_precision_overall = AverageMeter('Cls Precision', ':6.4f')
    cls_recall_overall = AverageMeter('Cls Recall', ':6.4f')
    cls_f1_overall = AverageMeter('Cls F1', ':6.4f')

    x_scale = configs.org_size[1]/configs.img_size[1]
    y_scale = configs.org_size[0]/configs.img_size[0]

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        for batch_idx, (batch_data, (_, labels, visibility, _)) in enumerate(tqdm(test_loader)):

            print(f'\n===================== batch_idx: {batch_idx} ================================')

            data_time.update(time.time() - start_time)
            batch_size = batch_data.size(0)

            batch_data = batch_data.to(configs.device)
            labels = labels.to(configs.device)
            labels = labels.float()
            # compute output

            if configs.model_choice == 'tracknet' or  configs.model_choice == 'tracknetv2' or configs.model_choice == 'wasb':
                # #for tracknet we need to rehsape the data
                B, N, C, H, W = batch_data.shape
                # Permute to bring frames and channels together
                batch_data = batch_data.permute(0, 2, 1, 3, 4).contiguous()  # Shape: [B, C, N, H, W]
                # Reshape to combine frames into the channel dimension
                batch_data = batch_data.view(B, N * C, H, W)  # Shape: [B, N*C, H, W]


            with torch.autocast(device_type='cuda'):
                output_heatmap, cls_score = model(batch_data.float()) # output in shape ([B, W],[B, H]) if output heatmap
      
        
            mse, rmse, mae, euclidean_distance = heatmap_calculate_metrics(output_heatmap, labels)
            post_processed_coords = extract_coords(output_heatmap)
            percision, recall, f1, accuracy = precision_recall_f1_tracknet(post_processed_coords, labels, distance_threshold=configs.ball_size)
            pred_x_logits, pred_y_logits = output_heatmap

            if cls_score == None:
                cls_accuracy = torch.tensor(1e-8, device=configs.device)
                cls_precision = torch.tensor(1e-8, device=configs.device)
                cls_recall = torch.tensor(1e-8, device=configs.device)
                cls_f1 = torch.tensor(1e-8, device=configs.device)
            else:
                cls_dict = classification_metrics(cls_score, visibility)
                cls_accuracy= cls_dict['accuracy']
                cls_precision = cls_dict['precision']
                cls_recall = cls_dict['recall']
                cls_f1= cls_dict['f1_score']



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
            percision_overall.update(percision)
            recall_overall.update(recall)
            f1_overall.update(f1)
            accuracy_overall.update(accuracy)
            cls_accuracy_overall.update(cls_accuracy)
            cls_precision_overall.update(cls_precision)
            cls_recall_overall.update(cls_recall)
            cls_f1_overall.update(cls_f1)

    print(
        'rmse_global: {:.4f}, real_rmse {:.4f}, Accuracy {:.4f}, Precision {:.4f}, recall {:.4f}, f1 {:.4f}, Classification Accuracy {:.4f}, Precision {:.4f}, recall {:.4f}, f1 {:.4f}'.format(rmse_overall.avg, real_rmse.avg, accuracy_overall.avg, percision_overall.avg, recall_overall.avg, f1_overall.avg, cls_accuracy_overall.avg, cls_precision_overall.avg, cls_recall_overall.avg, cls_f1_overall.avg))
    print('Done testing')


if __name__ == '__main__':
    main()
