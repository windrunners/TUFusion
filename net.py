import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange
from einops.layers.torch import Rearrange
import fusion_strategy


# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            out = F.relu(out, inplace=True)
        return out


# Dense convolution unit
class DenseConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseConv2d, self).__init__()
        self.dense_conv = ConvLayer(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.dense_conv(x)
        out = torch.cat([x, out], 1)
        return out


# CNN convolution unit
class DenseConv2d1(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseConv2d1, self).__init__()
        self.dense_conv1 = ConvLayer(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.dense_conv1(x)
        return out

# Resnet convolution unit
class ResnetBlock(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        super(ResnetBlock, self).__init__()
        out_channels_def = 16
        self.dense_conv_2_1 = ConvLayer(in_channels, out_channels_def, kernel_size, stride)
        self.dense_conv_2_2 = ConvLayer(in_channels, out_channels_def, kernel_size, stride)
        self.dense_conv_2_3 = ConvLayer(in_channels, out_channels_def*4, kernel_size, stride)


    def forward(self, x):
        out_1 = self.dense_conv_2_1(x)
        out_2 = self.dense_conv_2_2(out_1)
        out_3 = self.dense_conv_2_3(x+out_2)
        out = out_3

        return out

# Dense Block unit
class DenseBlock(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        super(DenseBlock, self).__init__()
        out_channels_def = 16
        denseblock = []
        denseblock += [DenseConv2d(in_channels, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def*2, out_channels_def, kernel_size, stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out

# CNN Block unit
class CNN(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        super(CNN, self).__init__()
        out_channels_def = 16

        denseblock3 = []
        denseblock3 += [DenseConv2d1(in_channels, out_channels_def*3, kernel_size, stride),
                       DenseConv2d1(in_channels+out_channels_def*2, out_channels_def*3, kernel_size, stride),
                       DenseConv2d1(in_channels+out_channels_def*2, out_channels_def*4, kernel_size, stride)]
        self.denseblock3 = nn.Sequential(*denseblock3)


    def forward(self, x):
        out = self.denseblock3(x)
        return out

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.inc = DoubleConv(1, 16)
        self.layer1 = DoubleConv(16, 32)
        self.layer2 = DoubleConv(32, 48)

    def forward(self, x, grads=None, name=None):
        x = self.inc(x)
        x = self.layer1(x)
        x = self.layer2(x)

        if grads is not None:
            x.register_hook(save_grad(grads, name + "_x"))
        return x


class Encoder_Trans(nn.Module):
    def __init__(self):
        super(Encoder_Trans, self).__init__()
        self.inc = DoubleConv(1, 16)
        self.layer1 = DoubleConv(17, 32)
        self.layer2 = DoubleConv(32, 48)
        self.transformer = ViT(image_size=256, patch_size=16, dim=256, depth=12, heads=16, mlp_dim=1024, dropout=0.1,
                               emb_dropout=0.1)

    def forward(self, x, grads=None, name=None):
        x_e = self.inc(x)
        x_t = self.transformer(x)
        x = torch.cat((x_e, x_t), dim=1)
        x = self.layer1(x)
        x = self.layer2(x)

        if grads is not None:
            x.register_hook(save_grad(grads, name + "_x"))
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = DoubleConv(48, 32)
        self.layer2 = DoubleConv(32, 16)
        self.outc = OutConv(16, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        output = self.outc(x)
        return output


class SimNet(nn.Module):
    def __init__(self):
        super(SimNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels=1, dim_head=64,
                 dropout=0., emb_dropout=0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim)
        )
        self.dim = dim
        self.patch_size = patch_size
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.convd1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, img):
        x = self.to_patch_embedding(img)  # [B,256,256]
        b, n, _ = x.shape

        x = self.transformer(x)
        x = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=self.patch_size, h=16, c=1)(x)  # [B,1,256,256]

        return x

# TUFusion network
class TUFusion_net(nn.Module):
    def __init__(self, input_nc=1, output_nc=1):
        super(TUFusion_net, self).__init__()
        denseblock = DenseBlock
        resnetblock = ResnetBlock

        nb_filter = [16, 65, 32, 16]
        kernel_size = 3
        stride = 1

        # encoder
        self.inc = DoubleConv(1, 63)
        self.conv1 = ConvLayer(input_nc, nb_filter[0], kernel_size, stride)
        self.DB1 = denseblock(nb_filter[0], kernel_size, stride)
        self.DB2 = resnetblock(nb_filter[0], kernel_size, stride)

        self.transformer = ViT(image_size=256, patch_size=16, dim=256, depth=12, heads=16, mlp_dim=1024, dropout=0.1,
                               emb_dropout=0.1)

        # decoder
        self.layer1 = DoubleConv(65, 32)
        self.layer2 = DoubleConv(32, 16)
        self.outc = OutConv(16, output_nc)

    def encoder(self, input):
        x1 = self.conv1(input)
        x_DB1 = self.DB1(x1)
        x_DB2 = self.DB2(x1)
        x_DB = (x_DB1 + x_DB2) / 2
        x_2 = self.transformer(input)
        x_3 = torch.cat((x_DB, x_2), dim=1)

        return [x_3]

    def fusion1(self, en1, en2):
        # add
        f_0 = (en1[0] + 1.5 * en2[0])/2
        return [f_0]

    def fusion(self, en1, en2, p_type):
        # hybrid
        # fusion_function = fusion_strategy.attention_fusion
        # f_fusion = fusion_function(en1[0], en2[0], p_type)

        # channel
        fusion_function = fusion_strategy.attention_fusion_channel
        f_fusion = fusion_function(en1[0], en2[0], p_type)

        # spatial
        # fusion_function = fusion_strategy.attention_fusion_spatial
        # f_fusion = fusion_function(en1[0], en2[0])
        return [f_fusion]

    def fusion2(self, en1, en2, p_type):
        fusion_function = fusion_strategy.attention_mechanism_fusion
        f_fusion = fusion_function(en1[0], en2[0], p_type)
        return [f_fusion]

    def decoder(self, en):
        x = self.layer1(en[0])
        x = self.layer2(x)
        output = self.outc(x)
        return [output]







