import torch
import os

def get_num_parameters(model):
    """Count number of trained parameters of the model"""
    if hasattr(model, 'module'):
        num_parameters = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
    else:
        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return num_parameters


def make_data_parallel(model, configs):
    if configs.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if configs.gpu_idx is not None:
            torch.cuda.set_device(configs.gpu_idx)
            model.cuda(configs.gpu_idx)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            configs.batch_size = int(configs.batch_size / configs.ngpus_per_node)
            configs.num_workers = int((configs.num_workers + configs.ngpus_per_node - 1) / configs.ngpus_per_node)

            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[configs.gpu_idx],
                                                              find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif configs.gpu_idx is not None:
        torch.cuda.set_device(configs.gpu_idx)
        model = model.cuda(configs.gpu_idx)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    return model

def post_process(coords_logits):
    # Assuming coords_logits is [B, 2] where each is a continuous logit, 
    # and we treat each dimension independently.

    x_coord_logits = coords_logits[:, 0]  # Shape: [B]
    y_coord_logits = coords_logits[:, 1]  # Shape: [B]

    # If you have only 2 values per sample, we need to expand them with classes
    # for each axis to use softmax correctly, otherwise `coords_logits` should be [B, W, H] or similar.

    # Use softmax for probabilistic interpretation
    x_coord_probs = torch.softmax(x_coord_logits, dim=-1)
    y_coord_probs = torch.softmax(y_coord_logits, dim=-1)

    # Use argmax to find the class (coordinate)
    x_coord_pred = torch.argmax(x_coord_probs, dim=-1)
    y_coord_pred = torch.argmax(y_coord_probs, dim=-1)

    # Stack the predictions to get [B, 2]
    return torch.stack([x_coord_pred, y_coord_pred], dim=1)


def load_pretrained_model(model, pretrained_path, gpu_idx):
    """Load weights from the pretrained model"""
    assert os.path.isfile(pretrained_path), "=> no checkpoint found at '{}'".format(pretrained_path)
    if gpu_idx is None:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
    else:
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(gpu_idx)
        checkpoint = torch.load(pretrained_path, map_location=loc)
    pretrained_dict = checkpoint['state_dict']
    
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
        
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                           if k in model_state_dict}
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() 
        #                    if k in model_state_dict and not k.startswith("class_head")}
        # 2. overwrite entries in the existing state dict
        model_state_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.module.load_state_dict(model_state_dict)
    else:
        model_state_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_state_dict}
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() 
        #                 if k in model_state_dict and not k.startswith("class_head")}
        # 2. overwrite entries in the existing state dict
        model_state_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_state_dict)
    return model
