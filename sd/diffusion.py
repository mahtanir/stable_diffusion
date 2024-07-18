import torch 
from torch import nn 
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention 

class TimeEmbedding(nn.Module):
    def __init__(self, n_embed) -> None:
        super().__init__()
        self.linear1 - nn.Linear(n_embed, 4*n_embed)
        self.linear_2 = nn.Linear(4*n_embed, 4*n_embed)
    def forward(self, x):
        #(1,320)
        x = self.linear1(x)
        x = F.silu(x)
        x = self.linear_2(x)

        #(1,1280)
        return x 
    

class SwitchSequential(nn.Sequential):
    def __init__(self, *args) -> None: #given x: latent (perhaps after passed?)
        super().__init__() #can also just do nn.sequential in init, put normal ones in self.encoders, and in forward
                            #of UNET just do self.encoders.children and then condition on isinstance. Benefit it simple and 
                            #don't have to rewrite all just rewrite once per instance as it appears in the nn.sequential
        self.layers = nn.ModuleList(args)
    def forward(self, x:torch.tensor, context: torch.tensor, time: torch.tensor):
        for layer in self.layers: #not exactly sure how this works check with gpt 
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)


class Upsample(nn.Module):
    def __init__(self, channels:int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        #(Batch size, channels, heigh, width) -> (batch size, features, height * 2, width * 2) 
        x = F.interpolate(x, scale_factor=2, mode="nearest") #same as nn.upsample used in VAE 
        #upsample just repeats the same input by the scale factor. i.e if 1 is first cells in put, will become [[1,1][1,1]] i.e 2*2 block. 
        return self.conv(x)
    

#Residual block is basically group norm followed by 2 convs (one initially no w,h change, but depth/channel/feature change
# and the last also has the inputted time information as well to ensure it's not lost (same params for each timestemp, makes it 
#easier for model to detach / control pixel values attributable to timestemp? ))
class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=1280) -> None:
        super().__init__()
        self.conv_features = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.merged_norm = nn.GroupNorm(32, out_channels)
        self.time_linear = nn.Linear(n_time, out_channels)
        self.residual_layer = None 
        if (in_channels == out_channels):
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, time):
        residual = x 
        #feature or x: Batch size, in_channels, height, width
        #time (1,1280)
        x = self.groupnorm_feature(x)
        x = F.silu(x)
        x = self.conv_features(x)
        time = F.silu(time)
        time = self.time_linear(time) #1,out_channels 
        x = x + time.unsqueeze(-1).unsqueeze(-1) #needs to be same dim. Adding time info to each pixel (SAME INFO to each)
        x = self.merged_norm(x)
        x = F.silu(x)
        x = self.conv_merged(x)
        x = x + self.residual_layer(residual)

class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head, n_embed, d_context = 768) -> None:
        super().__init__()
        channels = n_head * n_embed #number of channels outputted by UNETResidualBlock 
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels) #i.e how different from group norm if we specify channels? Auto calcs groups? 
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias = False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4*channels*2)
        self.linear_geglu_2 = nn.Linear(4*channels, channels)
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    def forward(self, x, context):
        #layer norm across pixels and features for each batch, but not across batches
        # x: (Batch size, channels, height, width)
        #context: (Batch Size, Seq_len, d_context Dim)
        residual_long = x 
        x = self.groupnorm(x)
        x = self.conv_input(x)  #no dimension changes at all 
        n, c, h, w = x.shape
        #(Batch size, channels, height, width) -> (Batch size, height *width, channels) so attention across for each pixel
        x = x.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3]).transpose(-1,-2)
        residual_short = x 
        x = self.layernorm_1(x) #i.e norm each pixel?
        x = self.attention_1(x)  #maintains same dimensions as the input by construction as long as passed n_embed = channels
        x = x + residual_short

        residual_short = x
        #(Batch size, height *width, channels)
        #Normalisation + Cross Attention With Skip Connection --. need to convert back to image from attention, HOW?
        x = self.layernorm_2(x)
        x = self.attention_2(x,context)
        x+= residual_short
        residual_short=x 


        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1) #4*channels*2 /2 now so 4*channels to channels
        x = x * F.gelu(gate) 

        x=self.linear_geglu_2(x)
        x += residual_short


        x = x.transpose(-1,-2)
        x = x.view((n,c,h,w))

        return self.conv_output(x) + residual_long

class UNET(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        #lhs first, encoders.
        self.encoders = nn.Sequential( #HERE SEQ MAKES SENSE 
            #Batch Size, 4, Height/8, Width/8 
            SwitchSequential(nn.Conv2d(4,320, kernel_size=3, padding=1)), #given a list of layers, will apply 1 by 1 but will know params?
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)), #n_head * n_embed = 320
            SwitchSequential(UNET_ResidualBlock(320,320), UNET_AttentionBlock(8,40)),
            #(Batch Size, 320, Height/8, Width/8) - > (Batch Size, 640, Height/16, Width/16 )
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, padding=1, stride=2)),
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(640,640), UNET_AttentionBlock(8,80)),
             #(Batch Size, 640, Height/16, Width/16) - > (Batch Size, 1280, Height/32, Width/32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, padding=1, stride=2)),
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1280,1280), UNET_AttentionBlock(8, 160)),
             #(Batch Size, 1280, Height/32, Width/32) - > (Batch Size, 1280, Height/64, Width/64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, padding=1, stride=2)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            SwitchSequential(UNET_ResidualBlock(1280,1280))

        )

        self.bottleneck =  SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160),
            UNET_ResidualBlock(1280, 1280)
        ) #bottom of unet i.e where necoder an ddecoder overlap

        self.decoders = nn.Sequential( #TODO: MAY HAVE TO CHANGE TO MODULE LIST, SEQ. MAKES NO SENSE!
            #(Batch size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)), #skip connection doubles the aount at each layer of the decoder for initial only 
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)), #double prev for initial channels
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),

            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),#n_head * n_embed = 640
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)), #n_head * n_embed = 640

            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)), #n_head * n_embed = 320
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),#n_head * n_embed = 640
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40))#n_head * n_embed = 320
        )
    def forward(self, x, context, time):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch Size, Seq Len, dim)
        #time: (1,1280)
        skip_connections = [] 
        for layer in self.encoders:
            x = layer(x, context, time) #context always separately considered. 
            skip_connections.append(x)
        
        x = self.bottleneck(x)
        
        #(Batch Size, 1280, h / 64, w / 64)
        for layers in self.decoders:
            x = torch.cat((x, skip_connections.pop()), dim=1) #i.e this is the skip connection from the corresponding encoder block. We concat to the channels. So it doubles hence the 2560 start 
            x = layers(x, context, time)

        return x
            
           
class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.groupNorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # (Batch size, 320, Height/ 8, Width/8)
        x = self.groupNorm(x)
        x = F.silu(x)
        x = self.conv(x)
        # (Batch size, 4, Height/ 8, Width/8)
        return x 

class Diffusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.time_embedding = TimeEmbedding(320) #320 = size of time embedding -> needs to know current timestemp - I guess embedding from srach 
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)
    
    def forward(self, latent: torch.tensor, context: torch.tensor, time: torch.tensor):
        #receives the latent from VAE, content from CLIP
        #latent: Batch size, 4, height/8, width / 8
        #context: (Batch_size, seq_len, d_embed)
        #Time: (1,320)

        #(1,320) -> (1,1280)
        time = self.time_embedding(time) #convert time to embedding -> positional encoding to convey info about time using sin and cosine like transformers. 

        #(Batch_size, 4, height/8, width/8) -> (Batch_size, 320, height/8, width/8)
        #More than start because : SD uses modified unet. Not like original which is the same. 
        output = self.unet(latent, context, time)

        #(Batch_size, 320, height/8, width/8) -> (Batch_size, 4, height/8, width/8) 
        output = self.final(output)

        # (Batch, 4, Height / 8, Width / 8)
        return output 
