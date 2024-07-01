import torch 
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock 

class VAE_Encoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1), #no change in dimensions 
            VAE_ResidualBlock(128, 128), #a series of ocnolutions nad normalisations 
            VAE_ResidualBlock(128, 128),
            #Batch size, channels, height, width --> Batch size, channels, height / 2, width 2
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0), 
            VAE_ResidualBlock(128, 256),
            VAE_ResidualBlock(256, 256),
            #Batch size , channels, height / 2, width / 2--> Batch size, channels, height / 4, width 4
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(256, 512),
            VAE_ResidualBlock(512, 512),
            #Batch size , channels, height / 4, width / 4--> Batch size, channels, height / 8, width 8
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512), #this basically allows for self-attention -> each pixel pays attention to the others (i.e not
            #just a pixel paing attention to the convolution pixels but across entire map.) i.e Each pixel is related to each other 
            #not independent or only thoughs within overlapping covolutions. 
            VAE_ResidualBlock(512, 512),
            nn.GroupNorm(32, 512), #32 groups https://towardsdatascience.com/what-is-group-normalization-45fe27307be7#:~:text=Group%20Normalization%20(GN)%20is%20a,group%20of%20channels%20as%20x%E1%B5%A2.
            nn.SiLU(), #a sigmoid like relu. 
            #Batch size , channels, height / 8, width / 8--> Batch size, 8, height / 8, width 8
            nn.Conv2d(512, 8, kernel_size=3, padding=1), #nochange in image dims 
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )

        #Why layer norm? 
    
    def forward(self, x: torch.tensor, noise: torch.tensor) -> torch.tensor:
        for layer in self.encoder.children():
            if ('stride' in layer and layer['stride'] == (2, 2)):
                x = F.pad(x, (0,1,0,1)) #we pad asymmetrically to ensure no information is losed, elise (n - 3 + 0) / 2 + 1 or n/2 - 1.5 + 1 
                # vs n - 3 + 1 (since padding asymmetric not 2p) / 2 + 1 so n /2. Despite floor, would lose info if you do the full iteration/foil of cals. 
            x = layer(x) 

            #sampling part Batch size, 8, height / 8, width 8
            mean, log_variance = torch.chunk(x, chunks=2, dim=1) #half of channels to represent std deviation for the encoder 
            #rep of an image, for each input. i.e perhaps pixel based generation per pixel 
            log_variance = torch.clamp(log_variance, -30, 20)
            variance = log_variance.exp()
            std = variance.sqrt()
            z = mean + std*noise #noise is epsilon 
            z *= 0.18215
            return z

        #in vae latent space is based on q(z|x) which approximates p(z|x) and is made to fit standardd normal gaussian prior
        # to impose space contraints and increase semantic meaning in the latent space I think. 
        #we are learning convolutions to map a mean and variance representation via sampling s.t for an input the convolutions 
        #map to unique mean and sigmas (i.e autoecnoder learns how to best map the mean and sigmas)
    
