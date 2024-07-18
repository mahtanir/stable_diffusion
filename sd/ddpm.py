#how do we remove the noise from the image.

import numpy as np 
import torch 

#beta schedule is the amount of noise at each timestemp 

class DDPMSampler:

    def __init__(self, generator:torch.Generator, num_training_steps=1000, beta_start: float = 0.000, beta_end: float = 0.0120):
        #forward process makes the image more noisy -> it adds noise to the image. Noise we add is dependenat on the variance schedule. beta_end is the one to make the image complete noise. 

        #linear schedule vs. cosine schedule. 
        self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_training_steps, dtype=torch.float32)**2 #returns equally spaced points from start to end. Why **0.5 then sq, same thing? 
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, 0) #(aplha_0, alpha_0*alpha_1, alpha_2*alpha_1*alpha_0)
        self.one = torch.tensor(1.0)

        self.generator = generator 
        self.num_training_steps = num_training_steps
        self.timestemps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())  #we are reversing because we start from a lot of noise 
        #we need to adjust timestemps based on num_training_steps 

    def set_inference_timestemps(self, num_inference_steps=50):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_training_steps // self.num_inference_steps
        timestemps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64) #this creates the actual timestemps.  --> this doesn't include num_inference-steps though...So will it ever be 100%? 
        self.timestemps = torch.from_numpy(timestemps)
    
    def _get_previous_timestemp(self, t):
        return t - self.num_training_steps // self.num_inference_steps
    
    #TODO: do we call ad noise before we reiterate / terurn the step.  -> no only inital to add that bit of noise at timestep 0 i.e complete noise. Then this equation defines the next markob chain step wherein the image itself 
    # will have some noise i.e x_{t-1} will have a lot of noise but less so, stc...  where noise added is defined by forward process as described in the add noise part, BUT here we are removing so its different. 
    # they are interconnected though, the derivation of backwards I'm pretty sure is reliant on forwards during bayes rule. i.e https://chatgpt.com/share/5ea02427-f66a-4254-a367-ed88f64e3b05
    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.tensor):
        t=timestep
        prev_t = self._get_previous_timestemp(t)

        alpha_prod_t = self.alpha_cumprod[timestep]
        alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t > 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev #i.e this is just alpha_t i.e 1 - beta_t 
        current_beta_t = 1- current_alpha_t

        #compute the predicted original sample x0 using formula 15 in DDPM paper 
        pred_original_sample = (latents - beta_prod_t**0.5 *model_output) / alpha_prod_t**0.5

        #formula 7 to get mean and variance to sample from dist for q(x_t-1 | x_t) 

        #step 1: compute the coefficients for the predicted original sample and the current sample x_t  (mean only) 
        pred_original_sample_coefficient = (alpha_prod_t_prev ** 0.5 * current_beta_t) / beta_prod_t
        current_sample_coefficient = current_alpha_t**0.5 * beta_prod_t_prev / (beta_prod_t)

        #compute the predicted previous sample mean 
        pred_prev_sample = pred_original_sample_coefficient * pred_original_sample + current_sample_coefficient*latents

        variance = 0 
        if t > 0:
            device = model_output.device 
            noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
            std_dev = (self._get_variance(t))**0.5 * noise #but get_variance here returns a single value right?
        
        pred_prev_sample = pred_prev_sample + std_dev
        return pred_prev_sample
    
    def set_strength(self, strength): #how much noise we want ot initially add to the image. The higher the strengt h the less flexibility we have because we're adjusting the number of iterations through the graph / inference steps 
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength) #i.e if strength is 80% then we skip 20% of the steps. So we don't start from 100% noise but rather 80% noise cause we skipped 20%. 
        #So it's less flexible the higher the strength is. 

        #I think we call num_inference_steps prior so timestemps only includes the revelant ones. Can then use the forward functions to figure out what the noisy image should be. I.e trick the unet that it created that noisy image for given timestemp. 
        self.timestemps = self.timestemps[start_step:] #i.e the add noise forward formula



    
    def _get_variance(self, timestemp: int) -> torch.Tensor:
        prev_t = self._get_previous_timestemp(timestemp)
        alpha_prod_t = self.alpha_cumprod[timestemp]
        alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one 
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        #formula 7 in ddpm paper
        variance =  (1 - alpha_prod_t_prev) / (1-alpha_prod_t) * current_beta_t
        variance = torch.clamp(variance, min=1e-20)
        return variance 
    
    def add_noise(self, original_samples: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor: #timesteps matter because the amount of noise we add becomes progressively less as we iterate (i.e most at t_max and least at t_min)
        alpha_cumprod = self.alpha_cumprod.to(original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        #this is the mean of q(xt | x0) 
        sqrt_alpha_prod = alpha_cumprod[timesteps] ** 0.5 
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod) < len(original_samples): #basically computing the the noisy images for each timestemp already using this provided equation. 
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)


        #in boradcasting usually it automatically adds one to the start so need to be a bit careful. Broadcasting works if dim the same or one of them is 1. Will add 1 to dim leading until shapes are the same. 
        #https://chatgpt.com/share/5ea02427-f66a-4254-a367-ed88f64e3b05

        #variance based on the equation provided in DDPM paper 
        sqrt_one_minus_alpha_prod = (1 - alpha_cumprod[timesteps]) ** 0.5 
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod) < len(original_samples):
            sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        #According to equation (4) of the DDPM paper 
        # Z = N(0, 1) -> N(mean, variance)=X?
        # X = mean + stdev * Z
        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)
        noisy_samples = (sqrt_alpha_prod*original_samples) + sqrt_one_minus_alpha_prod*noise
        return noisy_samples


        