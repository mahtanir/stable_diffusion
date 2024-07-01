import torch 
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention 

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    
    #first concert format to attention format, pass to self attention, then reshape back. 
    def forward(self, x: torch.tensor) -> torch.tensor: 
        # (Batch size, features, height, width)
        residual = x 
        n, c, h, w = x.shape
        x = x.view(n, c, h*w)
        #
        #(Batch size, features, height * width) -> (Batch size, height * width, features)
        x = x.transpose(-1, -2) #i.e each pixel now is defined by features i.e the channels
        #almost like each feature is a word with an embedding vector from which we calculate the attention mechanism. 
        x = self.attention(x) #often times I think multihead preserves the intial dimensions esp embedding dim i.e n_words or pixels * embedding dim
        x= x.transpose(-1,-2)
        x = x.view((n, c, h, w))
        return x + residual 



class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        
        self.group_norm1 = nn.GroupNorm(32, in_channels)
        self.silu = nn.SiLU()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.group_norm2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.residual_layer = nn.Identity()  #returns same input I believe 
        if (in_channels != out_channels):
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x):
        residual = x 
        x = self.group_norm1(x)
        x = self.silu(x)
        x = self.conv_1(x)
        x = self.group_norm2(x)
        x = self.silu(x)
        x = self.conv_2(x)
        x = x + self.residual_layer(residual)
        return x 
    
    #in nlp channels rep;laced by like sequence length, and since sentence length varies, will come a point toward longer semtemces where
    #not enough length in other sentences in batch for normalisation to be useful.

    #Basic idea is withouth normalisation the loss function oscillates too much making it hard to converge/training slower.

    class VAE_Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = nn.Sequential(
                nn.Conv2d(4,4, kernel_size=1, padding=0), #why not 8? -> cause we sample from std dev and mean which have 4 channels 
                #i.e batch, 4, height/8, width/8 i.e means std for each pixel it seems. 
                nn.Conv2d(4,512, kernel_size=3, padding=1),
                VAE_ResidualBlock(512, 512),
                VAE_AttentionBlock(512),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                #(Batch_size, 512, height/8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8) i.e no change yet
                VAE_ResidualBlock(512, 512),
                #(Batch_size, 512, height/8, Width / 8) -> (Batch_Size, 512, Height / 4, Width / 4)
                nn.Upsample(scale_factor=2), #perhaps alternative to convTranspose2d
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
               # (Batch_Size, 512, Height / 2, Width / 2)
                nn.Upsample(scale_factor=2),
                nn.Conv2d(512, 512, kernel_size=3, padding=1), 
                VAE_ResidualBlock(512, 256),
                VAE_ResidualBlock(256, 256),
                VAE_ResidualBlock(256, 256),
                #(Batch_Size, 512, Height , Width )
                nn.Upsample(scale_factor=2), #no learnable weights
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                VAE_ResidualBlock(256, 128),
                VAE_ResidualBlock(128, 128),
                VAE_ResidualBlock(128, 128),
                nn.GroupNorm(32, 128),
                nn.SiLU(), #usually follows norm of zs 
                nn.Conv2d(128, 3, kernel_size=3, padding=1)
            )
        
        def forward(self, x):
            # x: (Batch size, 4, Heigh/8, Width /8)
            x /= 0.18215
            #x: (Batch size, 3, height, width)
            x = self.decoder(x)
            return x 
