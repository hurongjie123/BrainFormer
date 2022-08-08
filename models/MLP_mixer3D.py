from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce

pair = lambda x: x if isinstance(x, tuple) else (x, x)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )

def MLPMixer(image_size, num_patches, dim, depth, num_classes, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
    #image_h, image_w = pair(image_size)
    #assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'


    #num_patches = 8*8*9
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )


class MLPMixer3D(nn.Module):

    def __init__(self, num_classes = 3):
        super().__init__()
        
        dim = 256
        patch_size = 16

        #self.patch_embedding = nn.Conv2d(1, dim, kernel_size=8, stride=8)
        self.patch_embedding = nn.Conv3d(1, dim, kernel_size=8, stride=8)
        
        self.MLPMixer = MLPMixer(
            image_size = 256,
            num_patches=8*8*6,
            dim = dim,
            depth = 12,
            num_classes = 3
        )
        #Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        #nn.Linear((patch_size ** 2) * channels, dim),


    def forward(self, x):

        #b, c, fh, fw, fd = x.shape
        #print('x1', x.size())
        x = self.patch_embedding(x)  # b,d,gh,gw
        #print('x2', x.size())
        #x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
        x = x.flatten(2).transpose(1, 2)
        #print('x3', x.size())
        x = self.MLPMixer(x)
       # print('x4', x.size())
        return x, x