import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import sys
sys.path.append('../')

from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from model.backbone_positional_encoding import build_backbone, create_positional_encoding
from model.temporal_transformer import build_transformer
from model.motion_model import build_motion_model
from utils.misc import inverse_sigmoid


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableBallDetection(nn.Module):
    def __init__(self, backbone, transformer, motion_model, num_queries, num_feature_levels, img_size, num_frames):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """

        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = transformer.d_model
        self.motion_model = motion_model

        self.coordinates_embed = nn.Linear(self.hidden_dim, 2)
        self.num_feature_levels = num_feature_levels
        self.sigmoid = nn.Sigmoid()

        # Learnable scalar weights for each frame in the sequence
        self.frame_weights = nn.Parameter(torch.ones(num_frames))  # Shape: [num_frames]
        
        # Feature extractor layers here...
        self.fc_x = nn.Linear(in_features=self.hidden_dim, out_features=img_size[1])  # Predicts x-coordinate
        self.fc_y = nn.Linear(in_features=self.hidden_dim, out_features=img_size[0])  # Predicts y-coordinate
        self.fc = nn.Linear(in_features=self.hidden_dim, out_features=2)  # where `aggregated_features_dim` is the hidden dimension size

        # set query embeddings
        self.query_embed = nn.Embedding(num_queries, self.hidden_dim*2)
        self.backbone = backbone

        # set input projection
        if num_feature_levels > 1:
            input_proj_list = []
            for _ in range(backbone.num_backbone_outputs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
            for _ in range(num_feature_levels - backbone.num_backbone_outputs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
                in_channels = self.hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([nn.Sequential(
                nn.Conv2d(backbone.out_channels, self.hidden_dim, kernel_size=1),
                nn.GroupNorm(32, self.hidden_dim),
            )])

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def combine_motion_featuremaps(self, motion_features, feature_maps):
        """
        Args:
            motion_features (tensor): [B, N-1, C, H, W]
            feature_maps (tensor): [B, N, C, H, W]
        Return:
            feature maps with motion information.
        """
        B, N, C, H, W = feature_maps.shape

        # Initialize a new tensor to store the combined results
        combined_feature_maps = torch.zeros_like(feature_maps)

        # Iterate over the frames (feature maps)
        for i in range(N):
            if i == 0:
                # Combine with the next frame's motion difference
                combined_feature_maps[:, i] = feature_maps[:, i] + motion_features[:, i]
            elif i == N - 1:
                # Combine with the previous frame's motion difference
                combined_feature_maps[:, i] = feature_maps[:, i] + motion_features[:, i - 1]
            else:
                # Combine with both previous and next frame's motion differences
                combined_feature_maps[:, i] = (feature_maps[:, i]
                                            + motion_features[:, i - 1]
                                            + motion_features[:, i])

        return combined_feature_maps


    def forward(self, samples):
        """ The forward expects a sample, which consists of:
               - samples.tensor: batched images represents frame , of shape [B, Number of pairs, 2, 3, H, W] or [B, Number of images, C, H ,W]

        
        """
        # first step is to split the tensor based on pair
        B, N, C, H, W = samples.shape
        # get motion features
        motion_features = self.motion_model(samples) # [B, N-1, C, H, W]
        # reshape to [B*N-1, C, H, W] then add extra dimension for the multiple feature level
        # motion_features = motion_features.view(B*(N-1), motion_features.size(2), motion_features.size(3), motion_features.size(4)).unsqueeze(dim=0)
        # get feature map from backbone
        # samples = samples.view(B*N, C, H, W) # [B*N, C, H, W]
        combined_samples = self.combine_motion_featuremaps(motion_features, samples).view(B*N, C, H, W)
        features_combined = self.backbone(combined_samples) # Expect it to output feature map which will be in shape [Number of feature level, B*N, C, H, W]
        # combine both motion and feature maps for better prediction'
        # features_reshaped = features.squeeze(0).view(B,N,features.size(2),features.size(3),features.size(4))
        # # print(f"motion features in shape {motion_features.shape}, feature maps in shape {features.shape}")
        # features_combined = self.combine_motion_featuremaps(motion_features, features_reshaped)
        # features_combined = features_combined.view(B*N, features_reshaped.size(2),features_reshaped.size(3),features_reshaped.size(4)).unsqueeze(0)
        srcs = []
        masks = []
        poses = []
        for layer, feat in enumerate(features_combined):
            projected_feature = self.input_proj[layer](feat)
            srcs.append(projected_feature)
            # all regions are valid so all pixels are 0
            mask = torch.zeros((B*N, *projected_feature.shape[-2:]), dtype=torch.bool, device=projected_feature.device)
            masks.append(mask)
            pos = create_positional_encoding(projected_feature)
            poses.append(pos)

        query_embeds = self.query_embed.weight # [8, 135, 512]
        # shape (1, [8, 512, 9, 15]), (1, [8, 9, 15]), (1, [8, 512, 9, 15]), ([1, 1024])
        # print(srcs[0].shape, masks[0].shape, poses[0].shape, query_embeds.shape)
        hs, init_reference, inter_references = self.transformer(srcs, masks, poses, query_embeds) # shape for hs is [B, number of queries, hidden dimension]'
        # print(hs.shape, init_reference.shape, inter_references.shape) #shape is torch.Size([B, N, 512]) torch.Size([7, 1, 2]) torch.Size([7, 1, 2])

        # Pass through the fully connected layers to produce logits for x and y coordinates
        x_coord_logits = self.fc_x(hs)  # [B, num_queries, w]
        y_coord_logits = self.fc_y(hs)  # [B, num_queries, h]

        # Option 1: Use maximum logit value as confidence
        x_max_logits, _ = torch.max(x_coord_logits, dim=-1)  # [B, num_queries]
        y_max_logits, _ = torch.max(y_coord_logits, dim=-1)  # [B, num_queries]
        confidence_scores = x_max_logits + y_max_logits      # [B, num_queries]

        # Get the index of the most confident query for each sample
        best_query_indices = torch.argmax(confidence_scores, dim=1)  # [B]

        batch_indices = torch.arange(B, device=hs.device)  # Ensure device matches
        # Index into x_coord_logits and y_coord_logits to get the best predictions
        x_coord_best = x_coord_logits[batch_indices, best_query_indices, :]  # [B, W]
        y_coord_best = y_coord_logits[batch_indices, best_query_indices, :]  # [B, H]


        # Option 2: If treating outputs as binary logits per coordinate, use sigmoid
        x_coord_probs = torch.sigmoid(x_coord_best)          # [B, W]
        y_coord_probs = torch.sigmoid(y_coord_best)          # [B, H]
                
        # Output the most confident coordinates
        output = (x_coord_probs, y_coord_probs)

        return output



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_detector(args):
    device = torch.device(args.device)

    chosen_feature_extractor = build_backbone(args)
    transformer = build_transformer(args)
    motion_model = build_motion_model(args)

    deformable_ball_detection_model = DeformableBallDetection(backbone=chosen_feature_extractor, transformer=transformer, 
                                                                motion_model=motion_model, num_queries=args.num_queries, 
                                                                num_feature_levels=args.num_feature_levels,
                                                                img_size=args.img_size, num_frames=args.num_frames).to(device)
    return deformable_ball_detection_model


if __name__ == '__main__':
    from config.config import parse_configs
    from data_process.dataloader import create_masked_train_val_dataloader
    device = 'cuda'
    configs = parse_configs()

    train_dataloader, val_dataloader, train_sampler = create_masked_train_val_dataloader(configs) 
    batch_data, (masked_frameids, masked_frames, labels) = next(iter(train_dataloader)) # batch data will be in shape [B, N, C, H, W]
    B, N, C, H, W = batch_data.shape
    data_reshaped = batch_data.to(device) # shape will be [B, N, C, H, W]
    data_reshaped = data_reshaped.float()
    print(f"data shape is {data_reshaped.shape}")

    deformable_ball_detection = build_detector(configs)


    out = deformable_ball_detection(data_reshaped)

    print(out[0].shape, out[1].shape)