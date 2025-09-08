import torch

class DDPMScheduler:

    def __init__(self, num_train_timesteps: int = 20000, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.num_train_timesteps = num_train_timesteps
        
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    @staticmethod
    def _get_tensor_for_t(tensor: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        batch_size = t.shape[0]
        out = tensor.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:

        sqrt_alphas_cumprod_t = self._get_tensor_for_t(self.sqrt_alphas_cumprod, timesteps, original_samples.shape)
        sqrt_one_minus_alphas_cumprod_t = self._get_tensor_for_t(self.sqrt_one_minus_alphas_cumprod, timesteps, original_samples.shape)

        noisy_samples = sqrt_alphas_cumprod_t * original_samples + sqrt_one_minus_alphas_cumprod_t * noise
        return noisy_samples

    def set_timesteps(self, num_inference_steps: int, device: torch.device):
        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.linspace(self.num_train_timesteps - 1, 0, num_inference_steps, device=device).long()

    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor) -> torch.Tensor:
        t = timestep

        alpha_t = self.alphas[t]
        beta_t = self.betas[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]


        pred_term = (sample - sqrt_one_minus_alpha_cumprod_t * model_output) / torch.sqrt(alpha_t)
        
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = torch.sqrt(beta_t) * noise 
            prev_sample = pred_term + variance
        else:
            prev_sample = pred_term
            
        return prev_sample