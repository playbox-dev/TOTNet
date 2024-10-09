import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import sys
sys.path.append('../')

from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from model.backbone_positional_encoding import build_backbone, create_positional_encoding
from model.transformer import build_transformer
from utils.misc import inverse_sigmoid


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableBallDetection(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels, img_size):
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
        self.class_embed = nn.Linear(self.hidden_dim, num_classes)
        self.bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
        self.coordinates_embed = nn.Linear(self.hidden_dim, 2)
        self.num_feature_levels = num_feature_levels
        self.sigmoid = nn.Sigmoid()
        
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
        

        # init constant bbox embedding layer 
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = transformer.decoder.num_layers
       
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
        self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
        self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
        self.transformer.decoder.bbox_embed = None
    



    def forward(self, samples):
        """ The forward expects a sample, which consists of:
               - samples.tensor: batched images represents frame , of shape [B, Number of pairs, 2, 3, H, W] or [B, Number of images, C, H ,W]

        
        """
        # first step is to split the tensor based on pair
        B, N, C, H, W = samples.shape
        samples = samples.view(B*N, C, H, W)
        features = self.backbone(samples) # Expect it to output feature map
        srcs = []
        masks = []
        poses = []
        for l, feat in enumerate(features):
            projected_feature = self.input_proj[l](feat)
            srcs.append(projected_feature)
            # all regions are valid so all pixels are 0 
            mask = torch.zeros((B*N, *projected_feature.shape[-2:]), dtype=torch.bool, device=projected_feature.device)
            masks.append(mask)
            pos = create_positional_encoding(projected_feature)
            poses.append(pos)

        query_embeds = self.query_embed.weight

        # Regarding transformer 
        # print(srcs.shape, masks.shape, pos.shape, query_embeds.shape)
        hs, init_reference, inter_references = self.transformer(srcs, masks, poses, query_embeds) # shape for hs is [B*N, number of queries, hidden dimension]
        hs_reshaped = hs.view(B, N, self.num_queries, self.hidden_dim)
        # print(hs_reshaped.shape, init_reference.shape, inter_references.shape) #shape is torch.Size([B, N, 512]) torch.Size([7, 1, 2]) torch.Size([7, 1, 2])

        hs_mean_frames = hs_reshaped.mean(dim=1)  # Shape [B, num_queries, hidden_dim]

        # Pass through the fully connected layers to produce logits for x and y coordinates
        x_coord_logits = self.fc_x(hs_mean_frames)  # [B, num_queries, w]
        y_coord_logits = self.fc_y(hs_mean_frames)  # [B, num_queries, h]

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



class Projection(nn.Module):
    def __init__(self, backbone_numchannels=2048, hidden_dim=512):
        super(Projection, self).__init__()  # Initialize the parent nn.Module

        self.input_proj = nn.Sequential(
            nn.Conv2d(backbone_numchannels, hidden_dim, kernel_size=1),
            nn.GroupNorm(32, hidden_dim),
        )
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Initializes the weights of the projection layers using Xavier uniform initialization
        and sets biases to zero.
        """
        xavier_uniform_(self.input_proj[0].weight, gain=1)
        if self.input_proj[0].bias is not None:
            constant_(self.input_proj[0].bias, 0)
        self.input_proj[1].weight.data.fill_(1)
        self.input_proj[1].bias.data.zero_()
    
    def forward(self, feature_maps):
        """
        Applies the projection layers to the input feature maps.

        Args:
            feature_maps (list of torch.Tensor): List of feature maps from the backbone.
                Each tensor should have shape [Batch * Pair number, C, H, W].

        Returns:
            list of torch.Tensor: List of projected feature maps with unified hidden_dim.
        """
        projected_feature = self.input_proj(feature_maps)
        return projected_feature


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
    

    deformable_ball_detection_model = DeformableBallDetection(backbone=chosen_feature_extractor, transformer=transformer, num_classes=args.num_classes, 
                                                        num_queries=args.num_queries, num_feature_levels=args.num_feature_levels, img_size=args.img_size).to(device)
    
    return deformable_ball_detection_model


if __name__ == '__main__':
    from config.config import parse_configs
    from data_process.dataloader import create_train_val_dataloader, create_masked_train_val_dataloader, draw_image_with_ball

    device = 'cuda'
    configs = parse_configs()
    # Create dataloaders
    # train_dataloader, val_dataloader, train_sampler = create_train_val_dataloader(configs)
    # batch_data, (masked_frameids, masked_frames, labels) = next(iter(train_dataloader))
    # # the dataloader exports data with shape [Batch, Num_of_pairs, num_images, C, H, W]
    # # where if we have total 9 frames, middle frame frame 5 will be masked, which leaves us 7 pairs they are
    # # [1,2], [2,3], [3,4], [4,6], [6,7], [7,8], [8,9] so for each batch it will contain this much of frames
    # # for example if we have batch 1, by suqeezing, we will have [7, 2, C,H, W], if batch is 8, we will have [56, 2, C, H, W]
    # B, num_pairs, num_images, C, H, W = batch_data.shape
    # data_reshaped = batch_data.view(B * num_pairs, num_images, C, H, W) # reshape to [B*N, 2, C, H, W]
    # # where on axis 1, if it is 0 it is the first frame in the pair, if it is 1, it is the second frame in the pair
    # frame_1 = data_reshaped[:, 0, :, :, :].view(B*num_pairs, C, H, W).to(device) # Extarct frame 1 and reshape to [B*P, C, H, W], which represents total number of frame 1
    # frame_1 = frame_1.float()



    train_dataloader, val_dataloader, train_sampler = create_masked_train_val_dataloader(configs) 
    batch_data, (masked_frameids, masked_frames, labels) = next(iter(train_dataloader)) # batch data will be in shape [B, 8, 3, H, W]
    B, num_images, C, H, W = batch_data.shape
    data_reshaped = batch_data.to(device) # shape will be [B*Number of images, 3, H, W]
    data_reshaped = data_reshaped.float()


    chosen_feature_extractor = build_backbone()
    transformer = Transformer(channels=2048, d_model=512, nhead=8, num_feature_levels=1).to(device)
    deformable_ball_detection = DeformableBallDetection(backbone=chosen_feature_extractor, transformer=transformer, num_classes=1, 
                                                        num_queries=1, num_feature_levels=1, aux_loss=False, with_box_refine=False).to(device)


    out = deformable_ball_detection(data_reshaped)
    # print(out)
    # print(labels, masked_frameids)
    # output_path=draw_image_with_ball(masked_frames[0], labels[0], "/home/s224705071/github/PhysicsInformedDeformableAttentionNetwork/src/model", 4)
    # print(output_path)