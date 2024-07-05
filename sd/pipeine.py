import torch 
import numpy as np 
from tqdm import tqdm 
from ddpm import DDPMSampler 

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8


#prompt is text prompt
#uncond prompt is text which we don't want to follow, usually just empty though 
# input_image is if we're passing an image to begin with, I think this is opposed to random noise 
# strength = ? 
#cfg_scale = classifer free giudance scale or how much attention to pay to the prompt 1--> 14
    #classifer free guidance: train a single network during training and with some probability we set the conditional signal to zero. The network becomes a mix of conditioned and uncodinition
    # (condition = with prompt) otherwise just a bunch of 0s for prompt in unconditioned. Not over reliant on prompt, can also produce p(x) vs p(x|c) or p(x,c)
#sampler name is ddpm 
#n_inference_steps I think is number of iterative denoising markov steps? 

def generate(prompt: str, uncond_prompt: str, input_image=None, strength=0.8, do_cfg=True, cfg_scale=7.5, sampler_name='ddpm', n_inference_steps=50, models={}, seed=None,
                device=None,
                idle_device=None, #i.e if model on cuda when idle can move to cpu 
                tokenizer=None
             ):
    #when we do inference, do torch.no_grad()
    with torch.no_grad():
        if (not (0 < strength <= 1)):
            raise ValueError('strength must be between 0 and 1')
        if idle_device:
            to_idle: lambda x: x.to(idle_device)
        else:
            to_idle: lambda x: x
        generator = torch.Generator(seed = None)
        if (seed is None): #generator used for random numbers? 
            # generate.seed() #TODO check this 
            generator.seed()
        else:
            generator.manual_seed(seed)
        
        clip = models['clip'] #passed into generate function 
        clip.to(device)

        if do_cfg:            #convert from torch.tensor to embedding vals with positional info AND then transformer attention on them with outer projection. 
            #convert the prompt into tokens using the tokenizer 
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=77).input_ids #make 77 length for sure
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            #(Batch Size = 1, Seq Len = 77) -> (Batch Size, Seq Len, Embed =768)
            cond_context = clip(cond_tokens)
            
            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding='max_length', max_length=77).input_ids #make 77 length for sure
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            uncond_context = clip(uncond_tokens)

            #(Batch Size = 2, Seq Len = 77, Embed = 768)
            context = torch.cat(cond_context, uncond_context)
        else: #I think prompt here is empty if no CFG or perhaps here it's the probability appraoch 
            tokens = tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=77).input_ids #make 77 length for sure
            tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            #(Batch Size = 1, Seq Len = 77) -> (Batch Size, Seq Len, Embed =768)
            tokens_context = clip(tokens)

        #samplers help to reduce the total number of steps/unet iterations to recursively take out noise. Different samplers are faster.
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps) 
        else:
            raise ValueError(f"Unkown sampler {sampler_name}")
        
        latents_shape = (1,4, LATENTS_HEIGHT, LATENTS_WIDTH) #i.e batch size = 1, 4 = from the chunk latent sampling mean and std, latent heights can be seen 512//8 from unet architecture

        if input_image: #i.e image to image with some text prompt to modify it slightly after imposing noise 
            encoder = models['encoder']
            encoder.to(device)


