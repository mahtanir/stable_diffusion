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
    # 2*(output_conditioned - output_unconditioned) + output_unconditioned where conditioning signal is the prompt and w is the cfg_scale 
#sampler name is ddpm  
#n_inference_steps I think is number of iterative denoising markov steps? 

# BASICALLY THE MAIN FUNCTION
# 1. Create tokenised and clip embedding for uncon and cond prompts/text. 
# 2. Create generator
# 3. If image is passed standardise and convert to latent passing it through encoder VAE else just make random noise of latent shape (generator random noise passed also of latent shape)
# 4. Create the ddpm sampler with the generator 
# 5. Add noise to the initial latent created 
# 6. For each timestep as per num_iterations, iterate through...
#   6.1 Create time embeddings 
#   6.2 If cfg, repeat the latent for batch size 2
#   6.3 Pass it into diffusion module or UNET to get the noise 
#   6.4 Remove the noise with the ddpm smapler from the noisy output and reiterate 
# 7. Pass it through the decoder 


def generate(prompt: str, uncond_prompt: str, input_image=None, strength=0.8 #how much attention we want to pay to the input image when we denoise it. i.e how much noise we want to add to it. 
             , do_cfg=True, cfg_scale=7.5, sampler_name='ddpm', n_inference_steps=50, models={}, seed=None,
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
            cond_context = clip(cond_tokens) #imposes attention on the text and embeds the position and the word embeddings as well. 
            
            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding='max_length', max_length=77).input_ids #make 77 length for sure
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            uncond_context = clip(uncond_tokens)

            #(Batch Size = 2, Seq Len = 77, Embed = 768)
            context = torch.cat(cond_context, uncond_context) #axis = 0 concat 
        else: #I think prompt here is empty if no CFG or perhaps here it's the probability appraoch or might just actually be the UNCODITIONAL PROMPT 
            tokens = tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=77).input_ids #make 77 length for sure
            tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            #(Batch Size = 1, Seq Len = 77) -> (Batch Size, Seq Len, Embed =768)
            tokens_context = clip(tokens)

        #samplers help to reduce the total number of steps/unet iterations to recursively take out noise. Different samplers are faster.
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator) #sampler that basically decides how much CONTROLLED noise to add for subsequent iteration but also how PRIMARILY much to remove from the current to get the new latent. 
            #It has a variance schedule for the noise. Such an eq for the removal and addition to create the new latent is based on the loss function and markov chain forward formula via this schedule / sampler. 
            sampler.set_inference_steps(n_inference_steps) 
        else:
            raise ValueError(f"Unkown sampler {sampler_name}")
        
        latents_shape = (1,4, LATENTS_HEIGHT, LATENTS_WIDTH) #i.e batch size = 1, 4 = from the chunk latent sampling mean and std, latent heights can be seen 512//8 from unet architecture

        #1. Reshape -> #2 Standardise 1-> -1 -> #3 nput it into the VAE encoder with randn noise generated from the random generator #4 -> add noise with the scheduler. 
        if input_image: #i.e image to image with some text prompt to modify it slightly after imposing noise 
            encoder = models['encoder']
            encoder.to(device)
            input_image_tensor = input_image.resize((HEIGHT, WIDTH)) #check the filetype here TODO
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
            input_image_tensor = rescale(input_image_tensor, (0,255), (-1,1)) #unet wants it normalised like this 
            #Height, Width, Channels -> Batch Size=1, Height, Width, Channels
            input_image_tensor = input_image_tensor.unsqueeze(0)
            #reshape for pytorch 
            input_image_tensor = input_image_tensor.permute(0,3,1,2)

            encoder_noise = torch.randn(latents_shape, generator=generator, device=device) #including noise for every pixel in the image. 
            #to the encoder 
            latents = encoder(input_image_tensor, encoder_noise) #the encoder_noise is different from the added noise to the image. Encoder noise is sampling noise to impose randomised latents being picked from a certain gaussain distribution. 
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timestemps[0])
            to_idle(encoder)
        else: #if just text -> image 
            #Start with random noise N(O, I)
            latents = torch.randn(latents.shape, generator=generator, device=device)
        
        diffusion = models["diffusion"]
        diffusion.to(device)

        #the ddpm sampler has timestemps. We just have 50 inference steps. I think the strength defines the timestep upper bound but it seems to go to 1000 regardless i.e total timesteps i.e strength = 999 then min = 0
        #for 1000, 1-> 1000. But since inference steps = 20, 1000->980 -> 960, etc...
        #Each of the timesteps indicates a noise level. 

        #PASSING IT INTO THE UNET 

        timestemps = tqdm(sampler.timesteps) #aren't the eq for x0 though so how are we using x_t 
        for i, timestep in timestemps:
            #(1,320) from an integer 
            time_embedding = get_time_embedding(timestep).to(device)
            model_input = latents #we've already created the latent but now we need to pass it into the UNET 

            if do_cfg: #Batch Size, Channels=4, Height=512//4, Width -> 2*Batch Size, Channels=4, Height, Width 
                model_input = model_input.repeat(2,1,1,1) #we need two latents for the cfg one conditioned and the other not ; i.e passing two latents into unet 
            
            #model_output is the predicted noise by the UNET 
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2, axis=0) #
                model_output = cfg_scale*(output_cond - output_uncond) + output_uncond
            
            #removing the noise predicted by the unet 
            latents = sampler.step(timestep, latents, model_output) #this is now used in latter iterations 
            #.step does... 1. Uses the predicted noise to update the latent for next iteration (does it add back the noise? )

        to_idle(diffusion)

        decoder = models['decoder']
        decoder.to(device)
        images = decoder(latents)
        to_idle(decoder)
        images = rescale(images, (-1,1), (0,255), clamp=True)
        #Convert to Batch Size, Height, Width,  Channels
        images = images.permute(0,2,3,1)
        images.to("cpu", torch.uint8).numpy()
        return images[0]
    
def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x = x - old_min
    x *= (new_max - new_min) / (old_max - old_min) #dividing makes it between 1 and 0 the old one, multiplying is for the new range. 
    x += new_min 
    if clamp:
        x = x.clamp(new_min, new_max)
    return x 
def get_time_embedding(timestemp):
    #we are using the positional encoding formulas here i.e the sin and cosine ones. 
    # (160, )
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) #we are not multiplying by 2? 
    # (1, 160)
    x = torch.tensor([timestemp], dtype=torch.float32)[:, None] * freqs[None] #torch.timestemp is the pos in the numerator i.e for eahc timestemp we have a position vector. AND first part is (1,1) col vector, second is (1,160) 
    #None indexing is actually a way to add a new column. [None] will add at the start i.e (a,) becomes (1,a) AND otherwise will add it where specified i.e for (a,b) [:,None] will become (a,1,b).
    #Works with numpy arrays and with tensors 
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)



#to understand how .step with the ddpm updates works.. 
# https://www.youtube.com/watch?v=HoKDTa5jHvg


        





