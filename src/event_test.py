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
from model.motion_model_light_event import build_motion_model_light
from model.wasb_event import build_wasb
from losses_metrics.metrics import heatmap_calculate_metrics, calculate_rmse, precision_recall_f1_tracknet, extract_coords, classification_metrics, classification_metrics_class_1
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

    
    if configs.model_choice == 'motion_light':
        model = build_motion_model_light(configs)
    elif configs.model_choice == 'wasb':
        model = build_wasb(configs)


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
    # Overall metrics
    rmse_overall = AverageMeter('RMSE_Overall', ':6.4f')
    accuracy_overall = AverageMeter('Accuracy', '6.4f')
    precision_overall = AverageMeter('Precision', '6.4f')
    recall_overall = AverageMeter('Recall', '6.4f')
    f1_overall = AverageMeter('F1', '6.4f')

    cls_accuracy_overall = AverageMeter('Cls Accuracy', ':6.4f')
    cls_precision_overall = AverageMeter('Cls Precision', ':6.4f')
    cls_recall_overall = AverageMeter('Cls Recall', ':6.4f')
    cls_f1_overall = AverageMeter('Cls F1', ':6.4f')

    # calculate fps rate
    total_time = 0
    total_frames = 0

    # switch to evaluate mode
    model.eval()
    start_time = time.perf_counter()  # Start timing
    with torch.no_grad():
        for batch_idx, (batch_data, (_, ball_xys, visibility, event_classes)) in enumerate(tqdm(test_loader)):
            print(f'\n===================== batch_idx: {batch_idx} ================================')
            batch_size = batch_data.size(0)
            batch_data = batch_data.to(configs.device)
            ball_xys = ball_xys.to(configs.device)
            event_classes = event_classes.to(configs.device)

            if configs.model_choice == 'wasb':
                B, N, C, H, W = batch_data.shape
                # Permute to bring frames and channels together
                batch_data = batch_data.permute(0, 2, 1, 3, 4).contiguous()  # Shape: [B, C, N, H, W]
                # Reshape to combine frames into the channel dimension
                batch_data = batch_data.view(B, N * C, H, W)  # Shape: [B, N*C, H, W]
           
            output_heatmap, cls_score = model(batch_data.float())

            _, rmse, _, _ = heatmap_calculate_metrics(output_heatmap, ball_xys)
            post_processed_coords = extract_coords(output_heatmap)

            precision, recall, f1, accuracy = precision_recall_f1_tracknet(
                post_processed_coords, ball_xys, distance_threshold=configs.ball_size
            )

            cls_dict = classification_metrics_class_1(cls_score, event_classes)
            cls_accuracy = cls_dict['accuracy']
            cls_precision = cls_dict['precision']
            cls_recall = cls_dict['recall']
            cls_f1 = cls_dict['f1_score']
          

            # Update metrics for each sample by visibility
            for sample_idx in range(batch_size):
               
                event_label = event_classes[sample_idx]  # Visibility label for this sample
                true_class = torch.argmax(event_label).item()
                label = ball_xys[sample_idx]

                pred_coords = post_processed_coords[sample_idx]
                pred_target = cls_score[sample_idx]
                predicted_class = torch.argmax(pred_target).item()

                x_pred, y_pred = pred_coords[0], pred_coords[1]

                # Compute RMSE for this sample
                sample_rmse = calculate_rmse(label[0], label[1], x_pred, y_pred)

                print('Real Event {}, Predict Event {} - Ball Detection - \t Overall: \t (x, y) - org: ({}, {}), prediction = ({}, {}), rmse is {}'.format(
                        true_class, predicted_class, label[0].item(), label[1].item(), x_pred.item(), y_pred.item(), sample_rmse))


            # Update overall metrics
            rmse_overall.update(rmse)
            accuracy_overall.update(accuracy)
            precision_overall.update(precision)
            recall_overall.update(recall)
            f1_overall.update(f1)

            cls_accuracy_overall.update(cls_accuracy)
            cls_precision_overall.update(cls_precision)
            cls_recall_overall.update(cls_recall)
            cls_f1_overall.update(cls_f1)


    end_time = time.perf_counter()  # End timing

    total_time += end_time - start_time  # Accumulate time
    total_frames = len(test_loader)*batch_size

    fps = total_frames / total_time if total_time > 0 else 0

    # Print results for each visibility category
    print("===== Specific Results =====")
    # Print overall results
    print(
        f"Overall Results: RMSE: {rmse_overall.avg:.4f}, Accuracy: {accuracy_overall.avg:.4f}, \n"
        f"Precision: {precision_overall.avg:.4f}, Recall: {recall_overall.avg:.4f}, F1: {f1_overall.avg:.4f}, \n"
        f"Cls Accuracy: {cls_accuracy_overall.avg:.4f}, Cls Precision: {cls_precision_overall.avg:.4f}\n"
        f"Cls Recall: {cls_recall_overall.avg:.4f}, Cls f1: {cls_f1_overall.avg:.4f}\n"
        f"Model fps rate: {fps:.0f}"
    )


if __name__ == '__main__':
    main()
