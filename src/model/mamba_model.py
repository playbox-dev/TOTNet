import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
sys.path.append('../')
from einops import rearrange, repeat


try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

from mamba_ssm import Mamba, Mamba2


class PatchMerging2D(nn.Module):
    """ Patch Merging Layer for 2D spatial patches.
    Args:
        input_resolution (tuple[int]): Resolution of input feature (H', W').
        dim (int): Number of input channels (feature dimension).
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution  # (H', W')
        self.dim = dim  # Input feature dimension (M)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)  # Reduce 4 * dim -> 2 * dim
        self.norm = norm_layer(4 * dim)  # Normalize concatenated features

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, N, M].
                N is the number of patches (H' * W').
        Returns:
            torch.Tensor: Merged patches with shape [B, T, N/4, 2*M].
        """
        B, T, N, M = x.shape
        H, W = self.input_resolution  # H' and W'

        # Ensure N matches H' * W'
        assert N == H * W, f"N ({N}) must match H' * W' ({H} * {W})"

        # Reshape N into 2D spatial grid (H', W')
        x = x.view(B, T, H, W, M)  # Shape: [B, T, H', W', M]

        #Pad spatial dimensions if not divisible by 2
        pad_h = H % 2
        pad_w = W % 2
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))  # Pad format: (C_left, C_right, W_left, W_right, H_top, H_bottom)


        # Extract 2x2 patches
        x0 = x[:, :, 0::2, 0::2, :]  # Top-left patches
        x1 = x[:, :, 1::2, 0::2, :]  # Bottom-left patches
        x2 = x[:, :, 0::2, 1::2, :]  # Top-right patches
        x3 = x[:, :, 1::2, 1::2, :]  # Bottom-right patches

        # Concatenate features from 2x2 patches
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # Shape: [B, T, H'/2, W'/2, 4*M]

        # Flatten spatial dimensions and normalize
        H_new = (H + pad_h) // 2
        W_new = (W + pad_w) // 2
        x = x.view(B, T, H_new * W_new, 4 * M)  # Shape: [B, T, N/4, 4*M]
        x = self.norm(x)

        # Reduce feature dimension
        x = self.reduction(x)  # Shape: [B, T, N/4, 2*M]

        return x, H_new * W_new

class PatchExpanding2D(nn.Module):
    """ Patch Expanding Layer for 2D spatial patches.
    Args:
        input_resolution (tuple[int]): Resolution of input feature (H', W').
        dim (int): Number of input channels (feature dimension).
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution  # (H', W')
        self.dim = dim  # Input feature dimension (M)
        self.expansion = nn.Linear(dim, dim*2, bias=False)  # Reduce dim -> dim*2
        self.norm = norm_layer(dim//2)  # Normalize before expansion

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, N, M].
                N is the number of patches (H' * W').
        Returns:
            torch.Tensor: Expanded patches with shape [B, T, 2*N, M/2].
        """
        B, T, N, M = x.shape
        H, W = self.input_resolution  # H' and W'

        # Ensure N matches H' * W'
        assert N == H * W, f"N ({N}) must match H' * W' ({H} * {W})"


        # Reduce feature dimensions
        x = self.expansion(x)  # Shape: [B, T, N, M*2]
        B, T, N, M = x.shape

        # Reshape N into 2D spatial grid (H', W')
        x = x.view(B, T, H, W, M)  # Shape: [B, T, H', W', M*2]
        H_new = H*2
        W_new = W*2
        out = rearrange(x, "b t h w (p1 p2 m)-> b t (h p1) (w p2) m", p1=2, p2=2, m=M//4)
        out = rearrange(out, "b t h w m -> b t (h w) m")

        # Normalize features
        out = self.norm(out)

        return out, H_new*W_new

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=(270, 480), patch_size=(15, 15), kernel_size=1, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.tubelet_size = kernel_size

        self.proj = nn.Conv3d(
            in_chans, embed_dim, 
            kernel_size=(kernel_size, patch_size[0], patch_size[1]),
            stride=(kernel_size, patch_size[0], patch_size[1])
        )

    def forward(self, x):
        x = self.proj(x)
        return x

class MambaBlock(nn.Module):
    def __init__(self, d_model, num_mamba_layers=2):
        """
        Args:
            d_model: Dimension of the input.
            num_mamba_layers: Number of Mamba layers in the block.
        """
        super().__init__()
        self.block = nn.Sequential(*[Mamba(d_model=d_model) for _ in range(num_mamba_layers)])
        self.bn = nn.BatchNorm1d(d_model)  # Use BatchNorm1d for sequence data
    
    def forward(self, x):
        """
        Forward pass for the Mamba block.
        Args:
            x: Input tensor of shape [B, N, M], where B is batch size, N is sequence length, and M is feature size.
        """
        x = self.block(x)  # Apply the Mamba layers
        x = rearrange(x, "b n m -> b m n")  # Rearrange for BatchNorm1d
        x = self.bn(x)  # Apply batch normalization
        x = rearrange(x, "b m n -> b n m")  # Rearrange back to original shape
        return x
    
class MambaEncoderPatchMergingBlock(nn.Module):
    def __init__(self, d_model, input_resolution, dim, num_mamba_layers=2):
        super().__init__()
        
        self.patch_merging = PatchMerging2D(input_resolution=input_resolution, dim=dim)
        self.mamba = MambaBlock(d_model=d_model, num_mamba_layers=num_mamba_layers)
        self.cls_token_proj = nn.Linear(dim, dim * 2)
        
    def forward(self, x, B, T, N):
        cls_tokens = x[:B, :1, :]
        x = x[:, 1:]
        x = rearrange(x, 'b (t n) m -> b t n m', t=T, n=N)  # [B, T, N, M]
        x, new_n = self.patch_merging(x) #[B, T, N//4, M*2]
        x = rearrange(x, 'b t n m -> b (t n) m')
        cls_tokens = self.cls_token_proj(cls_tokens)
        x = torch.cat((cls_tokens, x), dim=1) # append back onto the cls token 
        x = self.mamba(x) # still in same shape of [B, (T*N), M]

        return x, new_n

class MambaDecoderPatchExpandingBlock(nn.Module):
    def __init__(self, d_model, input_resolution, dim, num_mamba_layers=2):
        super().__init__()
        self.patch_expanding = PatchExpanding2D(input_resolution=input_resolution, dim=dim)
        self.mamba = MambaBlock(d_model=d_model, num_mamba_layers=num_mamba_layers)
        self.cls_token_proj = nn.Linear(dim, dim//2)
    
    def forward(self, x, B, T, N, skip_connection):
        cls_tokens = x[:B, :1, :]
        x = x[:, 1:]
        x = rearrange(x, 'b (t n) m -> b t n m', t=T, n=N)  # [B, T, N, M]
        x, new_n = self.patch_expanding(x)
        x = rearrange(x, 'b t n m -> b (t n) m')
        cls_tokens = self.cls_token_proj(cls_tokens)
        x = torch.cat((cls_tokens, x), dim=1) # append back onto the cls token 

        # skip connections 
        x = x + skip_connection
        x = self.mamba(x) # still in same shape of [B, (T*N), M]
        return x, new_n


class FlattenAndProject(nn.Module):
    def __init__(self, T_N, M):
        super().__init__()
        self.proj = nn.Linear(T_N * M, M)

    def forward(self, x):
        x = x.flatten(1, 2)  # Flatten T*N and M into a single dimension: [B, T*N*M]
        x = self.proj(x)     # Project to [B, M]
        return x



class MambaModel(nn.Module):
    def __init__(self, num_frames, img_size, patch_size=(5, 5), kernel_size=1, channels=3, embed_dim=96, drop_rate=0.2):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, 
            kernel_size=kernel_size,
            in_chans=channels, embed_dim=embed_dim
        )

        num_patches = self.patch_embed.num_patches

        # block 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, num_frames // kernel_size, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.block1 = MambaBlock(d_model=embed_dim)

        # block 2
        self.block2 = MambaEncoderPatchMergingBlock(d_model=embed_dim*2, input_resolution=(72, 128), dim=embed_dim)

        # block 3
        self.block3 = MambaEncoderPatchMergingBlock(d_model=embed_dim*4, input_resolution=(36, 64), dim=embed_dim*2)

        # bottleneck
        self.bottleNeck = MambaEncoderPatchMergingBlock(d_model=embed_dim*8, input_resolution=(18, 32), dim=embed_dim*4)

        # decoder block 4
        self.block4 = MambaDecoderPatchExpandingBlock(d_model=embed_dim*4, input_resolution=(9 ,16), dim=embed_dim*8)

        # decoder block 5
        self.block5 = MambaDecoderPatchExpandingBlock(d_model=embed_dim*2, input_resolution=(18 ,32), dim=embed_dim*4)

        # decoder block 6
        self.block6 = MambaDecoderPatchExpandingBlock(d_model=embed_dim, input_resolution=(36 ,64), dim=embed_dim*2)


        self.conv = nn.Sequential(
            nn.Conv3d(embed_dim, 64, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 1, kernel_size=(3, 3, 3), padding=1)
        )

        # self.final_proj = FlattenAndProject(T_N=num_frames*num_patches, M=embed_dim)
        self.proj_vertical = nn.Linear(embed_dim, img_size[0])  # Project to [B, H]
        self.proj_horizontal = nn.Linear(embed_dim, img_size[1])  # Project to [B, W]

        self.softmax = nn.Softmax(dim=-1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Initialize linear layers with Xavier initialization
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                # Initialize Conv2D layers with Kaiming initialization
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                # LayerNorm weights and biases are initialized to 1 and 0 respectively
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                # BatchNorm weights are initialized to 1, biases to 0
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        

    def forward(self, x):
        x = rearrange(x, 'b t c h w -> b c t h w')
        _, _, _, original_h, original_w = x.shape
        x = self.patch_embed(x)
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) (h w) c')

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed

        # temporal pos
        cls_tokens = x[:B, :1, :]
        x = x[:, 1:]
        _, N, M = x.shape
        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T) # n means number of patches, m means featre dimension 
        x = x + self.temporal_pos_embedding
        x = rearrange(x, '(b n) t m -> b (t n) m', b=B, t=T)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_drop(x)

        # block 1
        x = self.block1(x) # still in same shape of [B, (T*N), M]
        out1 = x.clone()
        # print(f"after block1 shape is {x.shape}, n is {N}")

        # block 2
        x, new_n = self.block2(x, B, T, N)
        out2 = x.clone()
        # print(f"after block2 shape is {x.shape}, n is {new_n}")

        # block 3
        x, new_n = self.block3(x, B, T, new_n)
        out3 = x.clone()
        # print(f"after block3 shape is {x.shape}, n is {new_n}")
    
        # bottleneck 
        x, new_n = self.bottleNeck(x, B, T, new_n)
        # print(f"after bottleneck shape is {x.shape}, n is {new_n}")

        # decoder block 4
        x, new_n = self.block4(x, B, T, new_n, out3)
        # print(f"after block4 shape is {x.shape}, n is {new_n}")

        # decoder block 5
        x, new_n = self.block5(x, B, T, new_n, out2)
        # print(f"after block5 shape is {x.shape}, n is {new_n}")

        # decoder block 6
        x, new_n = self.block6(x, B, T, new_n, out1)
        # print(f"after block6 shape is {x.shape}, n is {new_n}")

        # extract cls token
        cls_tokens = x[:B, :1, :]
        x = x[:, 1:] # x can be in shape [B, T*N, M]

        # project it to [B,M]
        # x = self.final_proj(x)

        # Max pooling across T*N
        # x = x.max(dim=1).values  # Output shape: [B, M]
        # Average pooling 
        x = x.mean(dim=1) 

        horizontal_heatmap = self.proj_horizontal(x)
        vertical_heatmap = self.proj_vertical(x)

        vertical_heatmap = self.softmax(vertical_heatmap)
        horizontal_heatmap = self.softmax(horizontal_heatmap) 

        return horizontal_heatmap, vertical_heatmap


def build_mamba(args):
    model = MambaModel(img_size=args.img_size, num_frames=args.num_frames).to(args.device)
    return model
    
if __name__ == '__main__':
    from model.model_utils import get_num_parameters
    from config.config import parse_configs
    from data_process.dataloader import create_occlusion_train_val_dataloader

    configs = parse_configs()
    configs.device = 'cuda'
    configs.num_frames = 5
    configs.img_size = (360, 640)
    configs.dataset_choice = 'tennis'

    train_dataloader, val_dataloader, train_sampler = create_occlusion_train_val_dataloader(configs)
    batch_data, (masked_frameids, labels, _, _) = next(iter(train_dataloader))
    batch_data = batch_data.to(configs.device)

    model = build_mamba(configs)

    print(f"motion model num params is {get_num_parameters(model)}")
    output = model(batch_data.float())
    print(output[0].shape, output[1].shape)
    print(torch.unique(output[0]), torch.unique(output[1]))
