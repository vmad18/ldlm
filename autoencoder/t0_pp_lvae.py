from contextlib import nullcontext
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import math

from einops import rearrange, repeat

from transformers import AutoTokenizer, PreTrainedTokenizerBase, PretrainedConfig, T5Config, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput

from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration

from .ae import VariationalAutoEncoder, Config, create_enc_dec_cfg
from contextlib import nullcontext
from typing import Tuple, Optional, Any

import re
import sys

class LatentEncoderConfig(Config): 
    
    dim: int = 768
    latent_dim: int = 256

    num_latents: int = 32 
    dim_head: int = 128
    
    max_tokens: int = 64 # 1024

    expansion_factor: int = 4

    use_rope: bool = True
    base: int = int(1e5)
    qk_norm: bool = False

    layers_p = 2
    dev = "cuda" if torch.cuda.is_available() else "cpu"


class LatentDecoderConfig(Config): 
    
    dim: int = 256
    latent_dim: int = 768

    num_latents: int = 64 # 1024 
    dim_head: int = 128
    
    max_tokens: int = 32

    expansion_factor: int = 4

    use_rope: bool = True
    base: int = int(1e5)
    qk_norm: bool = False

    layers_p = 2
    dev = "cuda" if torch.cuda.is_available() else "cpu"


class LatentVAEModel(T5ForConditionalGeneration): 

    def __init__(self,
                 config: T5Config, 
                 vocab_size: int, 
                 d_model: int = 768, 
                 latent_dim: int = 1024, 
                 num_latents: int = 32,
                 dim_head: int = 128, 
                 max_tokens = 2048, 
                 expansion_factor: int = 4, 
                 use_rope: bool = True, 
                 rope_base: int = int(1e5), 
                 qk_norm: bool = False, 
                 layers_p = 4,  
                 ctx = nullcontext(),
                 num_dev: int = 1,
                 dev: str = "cuda" if torch.cuda.is_available() else "cpu",
                 use_precomputed_latents: bool = False) -> None: 
        super().__init__(config)
        cfg_enc, cfg_dec = create_enc_dec_cfg(dim=d_model, 
                                              latent_dim=latent_dim, 
                                              num_latents=num_latents, 
                                              dim_head=dim_head,
                                              max_tokens=max_tokens, 
                                              expansion_factor=expansion_factor, use_rope=use_rope, 
                                              base=rope_base, qk_norm=qk_norm, layers_p=layers_p, dev=dev)
        self.t5_output_norm = nn.LayerNorm(config.d_model)

        if d_model == config.d_model:
            self.proj_down = nn.Identity() 
            self.proj_up = nn.Identity()
        else:
            self.proj_down = nn.Linear(config.d_model, d_model)
            self.proj_up = nn.Linear(d_model, config.d_model)

        self.vae = VariationalAutoEncoder(cfg_enc, cfg_dec)
        
        self.vocab_size = vocab_size
        self.dim = cfg_enc.dim
        self.latent_dim = cfg_enc.latent_dim
        self.num_latents = cfg_enc.num_latents

        self.max_tokens = cfg_enc.max_tokens
        self.num_dev = num_dev
        self.freeze = ctx
        
        self.use_precomputed_latents = use_precomputed_latents
        if self.use_precomputed_latents:
            # When using precomputed latents, the T5 encoder is not needed.
            # We delete it to save memory and speed up initialization.
            # We DO need the VAE's encoder (self.vae.encoder), which processes the latents.
            del self.encoder
            self.encoder = None

    def get_t5_encodings(self, input_ids: torch.Tensor, attn_mask: torch.Tensor):
        if self.encoder is None:
            raise ValueError("Cannot get T5 encodings when the model is in pre-computed latent mode.")
        with self.freeze:
            if self.num_dev > 1:
                encoder_outs = self.module.get_encoder()(input_ids = input_ids, attention_mask = attn_mask.bool())
            else:
                encoder_outs = self.get_encoder()(input_ids = input_ids, attention_mask = attn_mask.bool())

        return encoder_outs 


    def get_latents(self, input_ids: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor: 
        
        if attn_mask is None: 
            attn_mask = torch.ones_like(input_ids)

        encodings = self.get_t5_encodings(input_ids, attn_mask)[0]
        x = self.proj_down(encodings) 
        return self.vae.reparameterize(*self.vae.encode(x, attn_mask.bool()), only_mu=True)

    def latents_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Takes T5 embeddings and returns VAE latents (mu).
        This is for the diffusion training pipeline with precomputed T5 embeddings.
        """
        # Cast embeddings to the model's dtype to prevent mismatch
        embeddings = embeddings.to(self.dtype)

        projected_embeddings = self.proj_down(embeddings)
        moments = self.vae.encode(projected_embeddings)
        # For flow matching target, we only need the mean (mu) of the latent distribution
        return self.vae.reparameterize(*moments, only_mu=True)

    # decode diffused latent 
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latent)

    def autoencode(self, input_ids: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tuple[Any, dict]:
        *_, s = input_ids.shape 
        
        if attn_mask is None: 
            attn_mask = torch.ones_like(input_ids)

        encodings = self.get_t5_encodings(input_ids, attn_mask)  
        
        projected_encodings = self.proj_down(encodings[0])
        
        recon_encs, mu, log_var = self.vae(projected_encodings, attn_mask.bool())
        var_loss_dict = self.vae.cont_loss_func(recon_encs, projected_encodings, mu, log_var)
        
        recon_encs = self.proj_up(recon_encs[..., :s, :]) 
        encodings["last_hidden_state"] = recon_encs 
        
        return encodings, var_loss_dict['total_loss']

    def encode(self, latents: torch.Tensor) -> Tuple[Any, dict]:
        """
        Takes pre-computed latents, runs them through the VAE, and returns the VAE's output and loss.
        This is used when training only the VAE on pre-computed T5 latents.
        """
        projected_latents = self.proj_down(latents)
        recon_latents, mu, log_var = self.vae(projected_latents)
        var_loss_dict = self.vae.cont_loss_func(recon_latents, projected_latents, mu, log_var)
        return recon_latents, var_loss_dict['kld_loss']

    def decode_loss(self, original_latents: torch.Tensor, decoded_latents: Any) -> torch.Tensor:
        """
        Computes the reconstruction loss between original and decoded latents.
        This is primarily for the pre-computed latent training path.
        """
        # The VAE output might be a tuple or object, ensure we get the tensor
        if isinstance(decoded_latents, tuple):
            recon_latents = decoded_latents[0]
        else:
            recon_latents = decoded_latents
            
        recon_latents = self.proj_up(recon_latents)

        return F.mse_loss(recon_latents, original_latents)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        input_latents: Optional[torch.Tensor] = None,
        # This argument is for compatibility with generate()
        encoder_outputs: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:

        if encoder_outputs is not None:
            # This path is used by .generate()
            # Make sure to pass along any other kwargs .generate() might be using
            return super().forward(labels=labels, encoder_outputs=encoder_outputs, **kwargs)

        if input_latents is not None:
            # Pre-computed latent path

            # Cast input latents to the model's float dtype to prevent mismatch
            input_latents = input_latents.to(self.dtype)

            if not self.use_precomputed_latents:
                raise ValueError("`input_latents` was provided, but the model was not initialized with `use_precomputed_latents=True`.")
            
            recon_latents, kld_loss = self.encode(input_latents)
            recon_loss = self.decode_loss(input_latents, recon_latents)
            # For generation, we need to wrap the reconstructed latents in a BaseModelOutput
            projected_up_recon = self.proj_up(recon_latents)
            enc_outs = BaseModelOutput(last_hidden_state=projected_up_recon)
            return {
                'reconstruction_loss': recon_loss,
                'kld_loss': kld_loss,
                'encoder_outputs': enc_outs
            }

        if input_ids is not None:
            # On-the-fly tokenization path
            if self.use_precomputed_latents:
                raise ValueError("`input_ids` was provided, but the model was initialized with `use_precomputed_latents=True`.")
            enc_outs, vae_loss = self.autoencode(input_ids, attention_mask)
            lm_loss = super().forward(labels=labels, encoder_outputs=enc_outs, **kwargs).loss

            return {
                'lm_loss': lm_loss,
                'vae_loss': vae_loss,
                'encoder_outputs': enc_outs
            }

        raise ValueError("Either `input_ids` or `input_latents` must be provided.")


def get_latent_vae_tokenizer_t5(args, ctx, num_dev: int = 1, base_t5: str = "bigscience/T0pp") -> Tuple[LatentVAEModel, PreTrainedTokenizerBase, PretrainedConfig]:
    config = T5Config.from_pretrained(base_t5)
    tokenizer = T5Tokenizer.from_pretrained(base_t5)

    # Load a standard T5 model to get its pre-trained weights.
    # This is necessary for both the VAE-from-scratch path (to get a trained T5 encoder)
    # and the pre-computed latent path (to get a trained T5 decoder for generation).
    print(f"Loading pre-trained T5 weights from {base_t5}...")
    pretrained_t5 = T5ForConditionalGeneration.from_pretrained(base_t5)
    print("Pre-trained T5 model loaded.")

    print("\n" + "="*50)
    print("Initializing LatentVAEModel from scratch with the following parameters:")
    print(f"  d_model: {args.d_model}")
    print(f"  latent_dim: {args.latent_dim}")
    print(f"  num_latents: {args.num_latents}")
    print(f"  max_seq_len (as max_tokens): {args.max_seq_len}")
    print("="*50 + "\n")

    # Instantiate our custom model from scratch.
    # This ensures that our custom VAE components are initialized correctly,
    # avoiding the zero-initialization issue from using `from_pretrained` directly.
    vae_model = LatentVAEModel(
        config=config,
        vocab_size=len(tokenizer),
        d_model=args.d_model,
        latent_dim=args.latent_dim,
        num_latents=args.num_latents,
        dim_head=args.dim_head,
        max_tokens=args.max_seq_len,
        layers_p=args.num_layers,
        num_dev=num_dev,
        ctx=ctx,
        use_precomputed_latents=args.use_precomputed_latents
    )

    # Manually copy the pre-trained weights from the standard T5 model.
    # `strict=False` ensures that we only load weights for layers that match,
    # leaving our custom VAE weights with their correct random initialization.
    print("Copying pre-trained T5 weights into the LatentVAEModel...")
    vae_model.load_state_dict(pretrained_t5.state_dict(), strict=False)
    print("Weight copy complete.")

    # Clean up the reference model to free up memory
    del pretrained_t5

    if args.freeze_bb == 'ft':
        for (param_name, param) in vae_model.named_parameters():
            param.requires_grad = True
    elif args.freeze_bb == 'freeze':
        for (param_name, param) in vae_model.named_parameters():
            if re.fullmatch(".*vae.*", param_name):
                param.requires_grad = True
                # print(f"Trainable: {param_name}")
            else:
                param.requires_grad = False
    
    return vae_model, tokenizer, config  


if __name__ == "__main__": 
    model, token = get_latent_vae_tokenizer(nullcontext())

    x = torch.arange(100).view(2, 50)
    # model.bart_autoencode(x, torch.zeros_like(x), 1)
