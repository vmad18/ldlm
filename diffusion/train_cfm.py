from argparse import Namespace
from datetime import timedelta

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import AdamW

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from tqdm.auto import tqdm
from ema_pytorch import EMA

from transformers import get_scheduler, AutoTokenizer, PreTrainedTokenizerBase, T5ForConditionalGeneration, \
    MT5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bart.modeling_bart import BartForConditionalGeneration

from ldlm.diffusion.cond_flow_matcher import ConditionalFlowMatcher
from ldlm.autoencoder.t0_pp_lvae import get_latent_vae_tokenizer_t5
from ldlm.diffusion.neural_diffusion import DiTModel, DiTConfig

from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
import wandb

import os

import copy

from typing import Optional, Tuple

from pathlib import Path
from datetime import datetime
import json
import random

from ldlm.dataset_util.dataset_helper import get_dataset, get_dataloader_lvae_t5

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def get_output_dir(args):
    model_dir = f'{Path(args.dataset_name).stem}/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    output_dir = os.path.join(args.save_dir, model_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f'Created {output_dir}')
    return output_dir


def cycle(dl):
    while True:
        for data in dl:
            yield data


def exists(x):
    return x is not None


def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d


def euler_solver(x_0, t_steps: torch.Tensor, model: DiTModel, pbar=None):
    x = x_0
    dt = t_steps[1] - t_steps[0]

    for t_step in range(t_steps.shape[0] - 1):
        t = t_steps[t_step] * torch.ones(x_0.shape[0], device=x_0.device)
        dx = model(x, t)  # approx. trajectory step
        x = x + dx * dt  # euler integrate

        if pbar is not None:
            pbar.update(1)

    return x


def gen_samples(
        model: DiTModel,
        num_latents: int,
        dim_latents: int,
        batch_size: int,
        accelerator: Accelerator,
        step: int,  # current training step
        steps: int,  # number of gen steps
        method: str = "euler"):
    model.eval()

    with torch.no_grad():
        x_0 = torch.randn((batch_size, num_latents, dim_latents), device=accelerator.device)
        t_steps = torch.linspace(0, 1, steps, device=accelerator.device)

        if method == "euler":
            traj = euler_solver(x_0, t_steps, model, None)
        else:
            raise NotImplementedError

    # TODO save traj results to wandb

    model.train()


def compute_grad_norm(parameters):
    # implementation adapted from https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
    parameters = [p for p in parameters if p.grad is not None]
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), p=2) for p in parameters]), p=2).item()
    return total_norm


def separate_weight_decayable_params(params):
    # Exclude affine params in norms (e.g. LayerNorm, GroupNorm, etc.) and bias terms
    no_wd_params = [param for param in params if param.ndim < 2]
    wd_params = [param for param in params if param not in set(no_wd_params)]
    return wd_params, no_wd_params


def get_adamw_optimizer(params, lr, betas, weight_decay, eps=1e-8):
    params = list(params)
    wd_params, no_wd_params = separate_weight_decayable_params(params)

    param_groups = [
        {'params': wd_params},
        {'params': no_wd_params, 'weight_decay': 0},
    ]

    return AdamW(param_groups, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)


def ema(source: nn.Module, target: nn.Module, decay: float):
    """ EMA update """
    source_dict = source.state_dict()
    target_dict = target.state_dict()

    for key in source_dict.keys():
        target_dict[key].data.copy_(source_dict[key].data * decay + target_dict[key].data * (1 - decay))


class Trainer(object):

    def __init__(self,
                 args: Namespace,
                 gradient_accumulate_every=1,
                 init_lr=1e-4,
                 num_warmup_steps=500,
                 ema_decay=0.995,
                 adam_betas=(0.9, 0.99),
                 adam_weight_decay=0.01,
                 mixed_precision="no",
                 report_with: str = "wandb",
                 output_dir: str = "./results",
                 ) -> None:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_process_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=90))

        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulate_every,
            log_with=report_with,
            project_dir=output_dir,
            kwargs_handlers=[ddp_kwargs, init_process_kwargs],
        )

        # Determine the target dtype based on the mixed precision setting
        if self.accelerator.mixed_precision == "fp16":
            self.model_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.model_dtype = torch.bfloat16
        else:
            self.model_dtype = torch.float32

        self.output_dir = output_dir
        self.num_dev = self.accelerator.num_processes
        args.num_devices = self.num_dev

        self.save_and_sample_every = args.save_and_sample_every
        self.train_bs = args.train_bs
        self.eval_bs = args.eval_bs
        self.gradient_accumulate_every = gradient_accumulate_every
        self.ema_update_every = args.ema_update_every
        self.eval_every = args.eval_every

        # Load the arguments from the trained VAE model.
        # This ensures we build the VAE with the same architecture it was trained with.
        vae_args_path = os.path.join(args.latent_model_path, 'args.json')
        if not os.path.exists(vae_args_path):
            raise FileNotFoundError(f"VAE args.json not found at {vae_args_path}. Please ensure the latent_model_path is correct.")
        with open(vae_args_path, 'r') as f:
            vae_args = json.load(f)
        vae_args = Namespace(**vae_args)

        # When using the VAE to generate latents on the fly for diffusion training,
        # we must set this to False.
        vae_args.use_precomputed_latents = False

        # We also need to copy some arguments from the diffusion script args
        # to the VAE args, like the backbone freezing strategy.
        vae_args.freeze_bb = args.freeze_bb

        # Instantiate the T5-based VAE and its tokenizer
        self.ae, self.tokenizer, _ = get_latent_vae_tokenizer_t5(vae_args, torch.no_grad(), self.num_dev)

        # Load the trained VAE weights
        device = self.accelerator.device
        self.ae.to(device)
        data = torch.load(os.path.join(args.latent_model_path, 'model.pt'), map_location=device)
        
        # We use `strict=False` because the saved VAE model was trained without the T5 encoder
        # to save memory. The `get_latent_vae_tokenizer_t5` function has already loaded a
        # pre-trained T5 encoder for us. This command loads the trained VAE and decoder weights
        # from our saved file, while leaving the T5 encoder weights untouched.
        self.ae.load_state_dict(data['model'], strict=False)
        
        self.num_latents = self.ae.num_latents
        self.latent_dim = self.ae.latent_dim
        self.num_samples = args.num_samples 
        
        for param in self.ae.parameters():
            param.requires_grad = False

        self.ae.eval()

        if self.accelerator.is_main_process:
            if args.output_dir is None:
                args.output_dir = get_output_dir(args)
                with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
                    json.dump(args.__dict__, f, indent=2)
            results_folder = args.output_dir
            run = os.path.split(__file__)[-1].split(".")[0]
            if args.wandb_name:
                self.accelerator.init_trackers(run, config=args,
                                               init_kwargs={"wandb": {"dir": results_folder, "name": args.wandb_name}})
            else:
                self.accelerator.init_trackers(run, config=args, init_kwargs={"wandb": {"dir": results_folder}})

        self.fm = ConditionalFlowMatcher()

        cfg_dit = DiTConfig()
        cfg_dit.dim = args.model_dim
        cfg_dit.num_latents = self.ae.num_latents
        print(f"Setting cfg_dit.latent_dim: {self.latent_dim}")
        cfg_dit.latent_dim = self.latent_dim
        cfg_dit.num_layers = args.num_layers
        cfg_dit.dev = self.accelerator.device

        self.v_predictor = DiTModel(cfg_dit)
        # self.v_predictor.compile()

        self.ema_decay = ema_decay
        self.ema_model = copy.deepcopy(self.v_predictor)
        # self.ema_model.compile()

        self.opt = get_adamw_optimizer(self.v_predictor.parameters(), lr=init_lr, betas=adam_betas,
                                       weight_decay=adam_weight_decay)  # try to use muon?

        self.lr_scheduler = get_scheduler(
            args.lr_schedule,
            optimizer=self.opt,
            num_warmup_steps=num_warmup_steps * self.num_dev,
            num_training_steps=args.train_num_steps * self.num_dev,
        )

        self.dataset_name = args.dataset_name
        dataset = get_dataset(args.dataset_name, shard_size=args.shard_size)
        self.dataset = dataset.shuffle(seed=412)

        # The fineweb-edu dataset only has 'train' and 'valid' splits, so we remove 'test' logic.
        self.num_samples = min(self.num_samples, len(self.dataset['valid']))
        print(f'Using {self.num_samples} samples for evaluation')

        self.train_num_steps = args.train_num_steps

        # Use the more capable LVAE T5 dataloader
        self.dataloader = get_dataloader_lvae_t5(
            args, self.dataset['train'], self.ae.config, self.tokenizer,
            vae_args.max_seq_len,
            use_precomputed_latents=args.use_precomputed_latents,
            precomputed_latent_path=args.precomputed_latent_path,
            batch_size=self.train_bs
        )
        self.val_dataloader = get_dataloader_lvae_t5(
            args, self.dataset['valid'], self.ae.config, self.tokenizer,
            vae_args.max_seq_len, shuffle=False,
            use_precomputed_latents=args.use_precomputed_latents,
            precomputed_latent_path=args.precomputed_latent_path,
            batch_size=self.eval_bs
        )

        self.step = 0

        self.v_predictor, self.ema_model, self.opt, self.lr_scheduler, self.dataloader, self.val_dataloader = self.accelerator.prepare(
            self.v_predictor, self.ema_model, self.opt, self.lr_scheduler, self.dataloader, self.val_dataloader)

        model_size = sum(p.numel() for p in self.v_predictor.parameters())
        self.accelerator.print(f"Model params: {model_size / 1024 / 1024:.2f} M")

        self.data_iter = cycle(self.dataloader)
        self.val_iter = cycle(self.val_dataloader)

        self.reference_dict = {}
        self.use_precomputed_latents = args.use_precomputed_latents

        if self.accelerator.is_main_process:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(exist_ok=True)


    def save(self):
        if not self.accelerator.is_main_process:
            return
        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.v_predictor),
            'ema_model': self.accelerator.get_state_dict(self.ema_model),
            'opt': self.opt.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if self.accelerator.scaler is not None else None,
            'scheduler': self.lr_scheduler.state_dict()
        }

        torch.save(data, str(self.output_dir / f'model.pt'))

    def load(self, file_path=None, best=False, init_only=False):
        file_path = Path(file_path) if exists(file_path) else self.output_dir
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(file_path / f'model.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.v_predictor)
        ema_model = self.accelerator.unwrap_model(self.ema_model)
        model.load_state_dict(data['model'])

        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_local_main_process:
            ema_model.load_state_dict(data['ema_model'])

        self.step = data['step']

        if 'scheduler' in data:
            self.lr_scheduler.load_state_dict(data['scheduler'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        last_val_loss = float('nan') # Use Not a Number as a placeholder
        last_ema_val_loss = float('nan')
    
        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                total_loss = 0.
                
                # Using the dataloader directly is cleaner
                for batch in self.dataloader:
                    batch = {k: v.to(device) for k, v in batch.items()}

                    with accelerator.accumulate(self.v_predictor):
                        if self.use_precomputed_latents:
                            # The precomputed 'latents' are actually T5 embeddings.
                            # We need to pass them through the VAE's encoder to get the
                            # actual latents for the diffusion model.
                            t5_embeddings = batch['input_latents']
                            with torch.no_grad():
                                # Note: The attention mask is not available with precomputed latents.
                                # The `latents_from_embeddings` method will create a dummy mask of all ones.
                                latent = self.ae.latents_from_embeddings(t5_embeddings)
                        else:
                            with torch.no_grad():
                                latent = self.ae.get_latents(input_ids=batch['input_ids'],
                                                             attn_mask=batch['attention_mask'])
                        
                        # Cast latents to the diffusion model's dtype to prevent mismatch
                        latent = latent.to(dtype=self.model_dtype)

                        x_0 = torch.randn_like(latent)
                        t, x_t, u_t = self.fm.get_sample_location_and_conditional_flow(x_0, latent)

                        mask = None
                        v_t = self.v_predictor(x_t, mask, t)

                        loss = F.mse_loss(v_t.float(), u_t.float())
                        
                        # Accumulate loss for logging
                        total_loss += loss.detach() / self.gradient_accumulate_every
                        accelerator.backward(loss)
                    
                    # This block now runs only after accumulating gradients for the specified number of steps.
                    if accelerator.sync_gradients:
                        # Clip gradients
                        grad_norm = compute_grad_norm(self.v_predictor.parameters())
                        
                        accelerator.clip_grad_norm_(self.v_predictor.parameters(), 1.0)

                        # <<< FIX: Compute grad norm AFTER backprop and BEFORE optimizer step

                        # Optimizer step and scheduler
                        self.opt.step()
                        self.lr_scheduler.step()
                        self.opt.zero_grad()
                        
                        # EMA update
                        if (self.step + 1) % self.ema_update_every == 0:
                            ema(self.v_predictor, self.ema_model, self.ema_decay)

                        if accelerator.is_main_process:
                            logs = {
                                "loss": total_loss.item(),
                                "learning_rate": self.lr_scheduler.get_last_lr()[0],
                                "grad_norm": grad_norm, # Log the computed norm
                                "step": self.step,
                                "epoch": (self.step * self.gradient_accumulate_every) / len(self.dataloader),
                                "samples": self.step * self.train_bs * self.gradient_accumulate_every * self.num_dev, 
                            }
                            
                            # Validation Logic
                            if self.step > 0 and self.step % self.eval_every == 0:
                                self.v_predictor.eval()
                                self.ema_model.eval()

                                total_val_loss = 0.
                                total_ema_val_loss = 0.
                                
                                val_iter = iter(self.val_dataloader)
                                with torch.no_grad():
                                    for val_batch in val_iter:
                                        val_batch = {k: v.to(device) for k, v in val_batch.items()}
                                        if self.use_precomputed_latents:
                                            t5_embeddings = val_batch['input_latents']
                                            latent = self.ae.latents_from_embeddings(t5_embeddings)
                                        else:
                                            latent = self.ae.get_latents(input_ids=val_batch['input_ids'], attn_mask=val_batch['attention_mask'])
                                        
                                        # Cast latents to the diffusion model's dtype to prevent mismatch
                                        latent = latent.to(dtype=self.model_dtype)

                                        x_0 = torch.randn_like(latent)
                                        t, x_t, u_t = self.fm.get_sample_location_and_conditional_flow(x_0, latent)
                                        
                                        v_t = self.v_predictor(x_t, None, t)
                                        total_val_loss += F.mse_loss(v_t.float(), u_t.float()).item()

                                        v_t = self.ema_model(x_t, None, t)
                                        total_ema_val_loss += F.mse_loss(v_t.float(), u_t.float()).item()

                                last_val_loss = total_val_loss / len(self.val_dataloader)
                                last_ema_val_loss = total_ema_val_loss / len(self.val_dataloader)
                                
                                logs["val_loss"] = last_val_loss
                                logs["val_ema_loss"] = last_ema_val_loss
                                self.v_predictor.train()
                            
                            accelerator.log(logs, step=self.step)
                            logs["val_loss"] = last_val_loss
                            logs["val_ema_loss"] = last_ema_val_loss
                            pbar.set_postfix(**logs)
                            
                            
                        # Reset accumulated loss for the next set of accumulations
                        total_loss = 0.

                        # A training step is completed after the optimizer updates
                        self.step += 1
                        pbar.update(1)

                        if accelerator.is_main_process and self.step > 0 and self.step % self.save_and_sample_every == 0:
                            self.save()
                
        self.save()