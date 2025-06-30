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
import torch.distributed as dist

from pathlib import Path
from datetime import datetime
import json
import random

from ldlm.dataset_util.dataset_helper import get_dataset, get_dataloader_lvae_t5

__version__ = "0.0.1"

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
        t = ConditionalFlowMatcher.pad_t_like_x(t, x)
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
        steps: int,  # number of gen steps
        target_dtype: torch.dtype,
        method: str = "euler"):
    # The caller of this function should handle model.eval() and model.train()
    with torch.no_grad():
        x_0 = torch.randn((batch_size, num_latents, dim_latents), device=accelerator.device, dtype=target_dtype)
        t_steps = torch.linspace(0, 1, steps, device=accelerator.device)

        if method == "euler":
            traj = euler_solver(x_0, t_steps, model, None)
        else:
            raise NotImplementedError

    return traj


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

    @classmethod
    def from_pretrained_for_generation(cls, checkpoint_dir: str, mixed_precision: str = "bf16"):
        """
        Loads a pre-trained model from a checkpoint directory for generation.
        This is a factory method that constructs a trainer in "generation mode"
        without needing to instantiate any of the training-specific components.
        """
        checkpoint_dir = Path(checkpoint_dir)
        
        # --- 1. Load Training Args from Checkpoint ---
        # Be specific to avoid loading the vae.pt file by mistake
        model_files = list(checkpoint_dir.glob('model-*.pt')) + list(checkpoint_dir.glob('model.pt'))
        if not model_files:
            raise FileNotFoundError(f"No diffusion model checkpoint found in {checkpoint_dir}")
        model_checkpoint_path = max(model_files, key=os.path.getctime)
        print(f"Loading diffusion model checkpoint from: {model_checkpoint_path}")
        data = torch.load(model_checkpoint_path, map_location='cpu', weights_only=False)
        
        if 'args' not in data:
            raise ValueError(f"Checkpoint {model_checkpoint_path} does not contain 'args'. Cannot restore model.")
        args = data['args']

        # --- 2. Create a bare Trainer instance and Accelerator ---
        trainer = cls.__new__(cls)
        trainer.accelerator = Accelerator(mixed_precision=mixed_precision)

        # --- 3. Manually Set Up VAE and Tokenizer ---
        vae_args_path = checkpoint_dir / 'vae_args.json'
        if not vae_args_path.exists():
            raise FileNotFoundError(f"VAE args not found at {vae_args_path}. Checkpoint directory may be incomplete.")
        with open(vae_args_path, 'r') as f:
            trainer.vae_args = json.load(f)
        
        trainer.tokenizer = AutoTokenizer.from_pretrained("bigscience/T0pp", use_fast=False)
        vae_model_path = checkpoint_dir / 'vae.pt'
        if not vae_model_path.exists():
            raise FileNotFoundError(f"VAE model not found at {vae_model_path}. Checkpoint directory may be incomplete.")
        trainer.ae = torch.load(vae_model_path, map_location='cpu', weights_only=False)
        trainer.ae.to(trainer.accelerator.device)
        trainer.ae.eval()

        # --- 4. Manually Set Up Diffusion Model ---
        trainer.num_latents = trainer.ae.num_latents
        trainer.latent_dim = trainer.ae.latent_dim
        cfg_dit = DiTConfig()
        cfg_dit.dim = args.model_dim
        cfg_dit.num_latents = trainer.num_latents
        cfg_dit.latent_dim = trainer.latent_dim
        cfg_dit.num_layers = args.num_layers
        cfg_dit.dev = trainer.accelerator.device
        
        trainer.v_predictor = DiTModel(cfg_dit)
        trainer.ema_model = copy.deepcopy(trainer.v_predictor)

        # --- 5. Prepare Models and Set Generation Attributes ---
        trainer.v_predictor, trainer.ema_model = trainer.accelerator.prepare(trainer.v_predictor, trainer.ema_model)
        trainer.output_dir = checkpoint_dir
        trainer.step = 0 # Will be updated by load()

        if trainer.accelerator.mixed_precision == "fp16":
            trainer.model_dtype = torch.float16
        elif trainer.accelerator.mixed_precision == "bf16":
            trainer.model_dtype = torch.bfloat16
        else:
            trainer.model_dtype = torch.float32
        
        # --- 6. Load the Model Weights ---
        trainer.load(model_checkpoint_path)
        
        return trainer

    def __init__(self,
                 args: Namespace,
                 mixed_precision="no",
                 report_with: str = "wandb",
                 output_dir: str = "./results",
                 log_dir: str = './logs/',
                 ) -> None:
        # This __init__ method is now EXCLUSIVELY for training.
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_process_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=90))

        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=args.grad_accumulate,
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

        self.output_dir = os.path.join(output_dir, args.latent_model_path)
        self.num_dev = self.accelerator.num_processes
        args.num_devices = self.num_dev
        self.args = args

        self.save_and_sample_every = args.save_and_sample_every
        self.train_bs = args.train_bs
        self.eval_bs = args.eval_bs
        self.gradient_accumulate_every = args.grad_accumulate
        self.ema_update_every = args.ema_update_every
        self.eval_every = args.eval_every

        # generation args
        self.gen_steps = args.gen_steps
        self.gen_max_length = args.gen_max_length
        self.num_gen_samples = args.num_gen_samples
        self.gen_batch_size = args.gen_batch_size

        self.log_dir = log_dir
        self.use_precomputed_latents = args.use_precomputed_latents

        vae_args_path = os.path.join(args.latent_model_path, 'args.json')
        with open(vae_args_path, 'r') as f:
            self.vae_args = json.load(f)
        
        vae_args_ns = Namespace(**self.vae_args)
        self.ae, self.tokenizer, _ = get_latent_vae_tokenizer_t5(vae_args_ns, torch.no_grad(), self.num_dev, create_encoder=True)
        
        # Move the model to the correct device BEFORE loading the state dict
        self.ae.to(self.accelerator.device)

        # Load pre-trained weights if they exist
        vae_checkpoint_path = os.path.join(args.latent_model_path, 'vae.pt')
        if os.path.exists(vae_checkpoint_path):
            print(f"Loading existing VAE weights from {vae_checkpoint_path}...")
            
            # Load the checkpoint directly to the accelerator's device to avoid slow CPU overhead
            data = torch.load(vae_checkpoint_path, map_location="cpu", weights_only=False)
            
            # Handle both new (dictionary) and old (raw state_dict) formats
            if 'model_state_dict' in data:
                state_dict = data['model_state_dict']
            else:
                # Fallback for older checkpoints that were just the state_dict
                state_dict = data
            print("Loading state dict")
            self.ae.load_state_dict(state_dict)
            print("VAE weights loaded successfully.")
            del data # Free memory
            del state_dict

        self.ae.eval()

        self.num_latents = self.ae.num_latents
        self.latent_dim = self.ae.latent_dim
        self.num_samples = args.num_samples 
        
        for param in self.ae.parameters():
            param.requires_grad = False

        print("Starting Wandb")
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

        print("Creating ConditionalFlowMatcher")
        self.fm = ConditionalFlowMatcher()

        cfg_dit = DiTConfig()
        cfg_dit.dim = args.model_dim
        cfg_dit.num_latents = self.ae.num_latents
        print(f"Setting cfg_dit.latent_dim: {self.latent_dim}")
        cfg_dit.latent_dim = self.latent_dim
        cfg_dit.num_layers = args.num_layers
        cfg_dit.dev = self.accelerator.device

        self.v_predictor = DiTModel(cfg_dit)
        self.ema_decay = args.ema_decay
        self.ema_model = copy.deepcopy(self.v_predictor)

        self.opt = get_adamw_optimizer(self.v_predictor.parameters(), lr=args.init_lr, betas=(args.adam_beta1, args.adam_beta2),
                                       weight_decay=args.adam_weight_decay)

        self.lr_scheduler = get_scheduler(
            args.lr_schedule,
            optimizer=self.opt,
            num_warmup_steps=args.num_warmup_steps * self.num_dev,
            num_training_steps=args.train_num_steps * self.num_dev,
        )

        self.dataset_name = args.dataset_name
        dataset = get_dataset(args.dataset_name, shard_size=args.shard_size)
        self.dataset = dataset.shuffle(seed=412)

        self.num_samples = min(self.num_samples, len(self.dataset['valid']))
        print(f'Using {self.num_samples} samples for evaluation')

        self.train_num_steps = args.train_num_steps

        vae_args = Namespace(**self.vae_args) # Re-create for dataloaders
        self.dataloader = get_dataloader_lvae_t5(
            args, self.dataset['train'], self.ae.config, self.tokenizer,
            vae_args.max_seq_len,
            use_precomputed_latents=self.use_precomputed_latents,
            precomputed_latent_path=args.precomputed_latent_path,
            batch_size=self.train_bs
        )
        self.val_dataloader = get_dataloader_lvae_t5(
            args, self.dataset['valid'], self.ae.config, self.tokenizer,
            vae_args.max_seq_len, shuffle=False,
            use_precomputed_latents=self.use_precomputed_latents,
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

        if self.accelerator.is_main_process:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(exist_ok=True)


    def save(self, milestone):
        if not self.accelerator.is_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.v_predictor),
            'ema_model': self.accelerator.get_state_dict(self.ema_model),
            'opt': self.opt.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if self.accelerator.scaler is not None else None,
            'scheduler': self.lr_scheduler.state_dict(),
            'version': __version__,
            'args': self.args
        }

        model_save_path = os.path.join(self.output_dir, f'model-{milestone}.pt')
        torch.save(data, model_save_path)

        # Save the VAE model and its config to the same output directory
        vae_save_path = os.path.join(self.output_dir, 'vae.pt')
        torch.save(self.ae, vae_save_path)
        
        vae_args_save_path = os.path.join(self.output_dir, 'vae_args.json')
        with open(vae_args_save_path, 'w') as f:
            json.dump(self.vae_args, f, indent=2)

        print(f'saved model to {model_save_path}, VAE to {vae_save_path}, and VAE args to {vae_args_save_path}')

    def load(self, file_path=None, best=False, init_only=False):
        data = torch.load(file_path, map_location="cpu", weights_only=False)

        model = self.accelerator.unwrap_model(self.v_predictor)
        ema_model = self.accelerator.unwrap_model(self.ema_model)
        model.load_state_dict(data['model'])

          

        # self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_local_main_process:
            ema_model.load_state_dict(data['ema_model'])

        # move model to cuda
        model.to(self.accelerator.device)
        ema_model.to(self.accelerator.device) 

        self.step = data['step']

        # if 'scheduler' in data:
            # self.lr_scheduler.load_state_dict(data['scheduler'])

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
                                latent = self.ae.latents_from_embeddings(t5_embeddings)
                        else:
                            # When training from text, we run the text through the VAE to get the latents.
                            with torch.no_grad():
                                _, _, latent = self.ae.autoencode(input_ids=batch['input_ids'], attention_mask=batch.get('attention_mask'))
                        
                        # Cast latents to the diffusion model's dtype to prevent mismatch
                        latent = latent.to(dtype=self.model_dtype)

                        x_0 = torch.randn_like(latent)
                        t, x_t, u_t = self.fm.get_sample_location_and_conditional_flow(x_0, latent)

                        v_t = self.v_predictor(x_t, t)

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
                                num_val_batches = 10  # Only evaluate on 10 batches
                                
                                with torch.no_grad():
                                    for _ in range(num_val_batches):
                                        try:
                                            val_batch = next(val_iter)
                                        except StopIteration:
                                            break
                                            
                                        val_batch = {k: v.to(device) for k, v in val_batch.items()}
                                        if self.use_precomputed_latents:
                                            t5_embeddings = val_batch['input_latents']
                                            with torch.no_grad():
                                                latent = self.ae.latents_from_embeddings(t5_embeddings)
                                        else:
                                            with torch.no_grad():
                                                _, _, latent = self.ae.autoencode(input_ids=val_batch['input_ids'], attention_mask=val_batch.get('attention_mask'))
                                        
                                        # Cast latents to the diffusion model's dtype to prevent mismatch
                                        latent = latent.to(dtype=self.model_dtype)

                                        x_0 = torch.randn_like(latent)
                                        t, x_t, u_t = self.fm.get_sample_location_and_conditional_flow(x_0, latent)
                                        
                                        v_t = self.v_predictor(x_t, t)
                                        total_val_loss += F.mse_loss(v_t.float(), u_t.float()).item()

                                        v_t = self.ema_model(x_t, t)
                                        total_ema_val_loss += F.mse_loss(v_t.float(), u_t.float()).item()

                                last_val_loss = total_val_loss / num_val_batches
                                last_ema_val_loss = total_ema_val_loss / num_val_batches
                                
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
                            self.save(self.step)
                            self.eval(
                                batch_size=self.eval_bs,
                                gen_mult=self.gen_steps,
                                max_gen_length=self.gen_max_length,
                                num_samples=self.num_gen_samples,
                            )
                

    @torch.no_grad()
    def eval(self,
             batch_size: int = 1,
             gen_mult: int = 1,
             max_gen_length: int = 512,
             num_samples: int = 1) -> None:
        """
        Evaluate the model by generating samples and calculating perplexity.
        """
        accelerator = self.accelerator
        # It's best practice to use the EMA model for generation.
        model_for_generation = self.accelerator.unwrap_model(self.ema_model)
        model_for_generation.eval()
        self.ae.eval()

        generated_texts = []
        pbar = tqdm(range(0, num_samples, batch_size), desc=f"Generating {num_samples} samples")

        for i in pbar:
            batch_size = min(batch_size, num_samples - i)
            
            # Generate latents using the diffusion model
            latents = gen_samples(
                model=model_for_generation,
                num_latents=self.num_latents,
                dim_latents=self.latent_dim,
                batch_size=batch_size,
                accelerator=self.accelerator,
                steps=gen_mult,
                target_dtype=self.model_dtype, # Pass the correct dtype for noise generation
                method="euler"
            )
            # Decode the latents into text using the VAE
            with torch.no_grad():
                output_ids = self.ae.decode_latent(latents)
            
            decoded_batch = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            generated_texts.extend(decoded_batch)

            # Print samples as they are generated
            for j, text in enumerate(decoded_batch):
                print(f"\n--- Sample {i+j+1} ---")
                print(text)
                print("-" * (16 + len(str(i+j+1))) + "\n")

            # Log to wandb in batches if enabled
            if "wandb" in self.accelerator.log_with:
                try:
                    table = wandb.Table(columns=["step", "sample_id", "generated_text"])
                    for j, text in enumerate(decoded_batch):
                        table.add_data(self.step, i+j, text)
                    self.accelerator.log({"generated_samples": table}, step=self.step)
                except ImportError:
                    print("wandb not installed, skipping logging of generated samples.")

            pbar.set_description(f"Generated {len(generated_texts)} samples")
        print("\nEvaluation finished.")