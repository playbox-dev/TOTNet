import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import copy
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

sys.path.append('../')
from model.ops.modules.ms_deform_attn import DeformAttn
from model.backbone_positional_encoding import create_positional_encoding
from model.temporal_model import PixelTemporalSelfAttention, TemporalDeformableSelfAttentionEncoder
from utils.misc import inverse_sigmoid

class Transformer(nn.Module):
    def __init__(self, channels=2048, d_model=512, nhead=8, num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=2048, 
                 dropout=0.1, activation="relu", num_feature_levels=4, enc_n_points=4, num_frames=9, batch_size=8):
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_frames = num_frames
        self.embedder_layer = nn.Linear(channels, d_model)
        self.batch_size = batch_size
        encoder_layer = EncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = Encoder(encoder_layer, num_encoder_layers)

        # temporal model
        # self.temporal_model = PixelTemporalSelfAttention(self.d_model, self.nhead, num_encoder_layers)
        # self.temporal_model = TemporalDeformableSelfAttentionEncoder(d_model=d_model, d_ffn=dim_feedforward, dropout=dropout,
        #                                                              activation=activation, n_levels=num_encoder_layers)

        decoder_layer = DecoderLayer(d_model=d_model, d_ffn=dim_feedforward, 
                                     dropout=dropout, activation=activation, n_levels=num_feature_levels, n_heads=nhead, n_points=enc_n_points)
        self.decoder = Decoder(decoder_layer=decoder_layer, num_layers=num_decoder_layers)

        self.reference_points = nn.Linear(d_model, 2)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, DeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    
    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio


    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        """
        forward function for the transformer
        Args:
            srcs(list): [lvl, B*P, hidden_dimension, H, W]
            masks(list): [lvl, B*P, H, W]
            pos_embeds(list): [lvl, 1, hidden_dimension, H, W]
            query_embed(tensor): [num_queries, hidden_dimension]
        Return: 
            hs(tensor): hidden state in shape [B*P, number of queries, hidden_dimension]
        """
        assert query_embed is not None, "query_embedding is none"
   
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(mask) for mask in masks], dim=1)  # Correct stacking over batch size
        
        # src_flatten.shape [B*N, C, Hidden_dimension], spatial_shapes shape [num_feature_level, 2] lvl_pos_embed_flatten.shape [1, C, Hidden_dimension]
        
        mask_flatten = None 
        # memory shape [B*N, H*W, d_model]
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        bs, HW, c = memory.shape
        # print(f"memory shape is {memory.shape}")
        B, N = self.batch_size, self.num_frames
        memory = memory.view(B, N, HW, c)
        # go through temporal model which is deformable temporal model where we utilize other frame information
        # memory = self.temporal_model(memory) # this will output shape [B, HW, C]
        memory = memory[:, -1, :, :] # make it [B, HW, C]

        query_embed, tgt = torch.split(query_embed, c, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(B, -1, -1)
        tgt = tgt.unsqueeze(0).expand(B, -1, -1)
        reference_points = self.reference_points(query_embed).sigmoid() # shape [B, num_queries, 2]
        valid_ratios = valid_ratios[:B, :, :]
        init_reference_out = reference_points
    
        hs, inter_references = self.decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten)
        
        inter_references_out = inter_references

        return hs, init_reference_out, inter_references_out
    



class EncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        """
        Args:
            d_model: the dimensionality of the embeddings used throughout the model
        """

        # self attention
        self.self_attn = DeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
    
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        """
        Args:

        """

        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src

class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(Encoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
    
    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        
            # remove since all valid
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
  
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        """
        args:
            src(tensor): [B*N, C, Hidden_dimmension]
            spatial_shapes(tensor): [Number of feature level, 2]
            level_start_index(tensor): []
            valid_ratios: [B*N, Number of feature level, 2]
        """
        output = src
        # references shape [B*N, HW, 1 2]
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)

        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class DecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = DeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def get_valid_ratio(mask):
    _, H, W = mask.shape
    valid_H = torch.sum(~mask[:, :, 0], 1)
    valid_W = torch.sum(~mask[:, 0, :], 1)
    valid_ratio_h = valid_H.float() / H
    valid_ratio_w = valid_W.float() / W
    valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
    return valid_ratio


def build_transformer(args):
    if args.distributed:
        batch_size = torch.tensor(args.batch_size // torch.cuda.device_count(), dtype=torch.int32)
    else:
        batch_size = torch.tensor(args.batch_size, dtype=torch.int32)
    transformer = Transformer(channels=args.backbone_out_channels, d_model=args.transfromer_dmodel, nhead=args.transformer_nhead, 
                              num_feature_levels=args.num_feature_levels, num_frames=args.num_frames, batch_size=batch_size).to(args.device)
    return transformer


if __name__ == '__main__':
    from config.config import parse_configs
    from data_process.dataloader import create_masked_train_val_dataloader
    device = 'cuda'
    configs = parse_configs()


    transformer=build_transformer(configs)
    dummy_input = [torch.randn([72, 512, 9, 15], device=device, dtype=torch.float32)]
    masks = [torch.zeros([72, 9, 15], device=device, dtype=torch.int)]
    pos_embedding = [torch.randn([72, 512, 9, 15], device=device, dtype=torch.float32)]
    query_embedding = torch.randn([1, 1024], device=device, dtype=torch.float32)

    output, _, _ = transformer(dummy_input, masks, pos_embedding, query_embedding)
    print(f"final output shape {output.shape}")