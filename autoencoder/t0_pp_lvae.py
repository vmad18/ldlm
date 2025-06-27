from contextlib import nullcontext
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from einops import rearrange, repeat

from transformers import AutoTokenizer, PreTrainedTokenizerBase, PretrainedConfig, T5Config, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput

from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration

from .ae import VariationalAutoEncoder, Config, create_enc_dec_cfg
from contextlib import nullcontext
from typing import Tuple, Optional, Any

import re

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
                 dev: str = "cuda" if torch.cuda.is_available() else "cpu",) -> None: 
        super().__init__(config)

        cfg_enc, cfg_dec = create_enc_dec_cfg(dim=d_model, 
                                              latent_dim=latent_dim, 
                                              num_latents=num_latents, 
                                              dim_head=dim_head,
                                              max_tokens=max_tokens, 
                                              expansion_factor=expansion_factor, use_rope=use_rope, 
                                              base=rope_base, qk_norm=qk_norm, layers_p=layers_p, dev=dev)

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

    
    def get_t5_encodings(self, input_ids: torch.Tensor, attn_mask: torch.Tensor):
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
    ) -> dict:

        if encoder_outputs is not None:
            # This path is used by .generate()
            return super().forward(labels=labels, encoder_outputs=encoder_outputs)

        if input_latents is not None:
            # Pre-computed latent path
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
            enc_outs, vae_loss = self.autoencode(input_ids, attention_mask)
            lm_loss = super().forward(labels=labels, encoder_outputs=enc_outs).loss

            return {
                'lm_loss': lm_loss,
                'vae_loss': vae_loss,
                'encoder_outputs': enc_outs
            }

        raise ValueError("Either `input_ids` or `input_latents` must be provided.")


def get_latent_vae_tokenizer_t5(args, ctx, num_dev: int = 1, base_t5: str = "bigscience/T0pp") -> Tuple[LatentVAEModel, PreTrainedTokenizerBase, PretrainedConfig]:
    config = T5ForConditionalGeneration.from_pretrained(base_t5).config 
    tokenizer = T5Tokenizer.from_pretrained(base_t5)

    print("\n" + "="*50)
    print("Initializing LatentVAEModel with the following parameters:")
    print(f"  d_model: {args.d_model}")
    print(f"  latent_dim: {args.latent_dim}")
    print(f"  num_latents: {args.num_latents}")
    print(f"  max_seq_len (as max_tokens): {args.max_seq_len}")
    print("="*50 + "\n")

    vae = LatentVAEModel.from_pretrained(
        base_t5,
        config = config, 
        vocab_size = len(tokenizer),
        d_model = args.d_model, 
        latent_dim = args.latent_dim, 
        num_latents = args.num_latents, 
        dim_head = args.dim_head, 
        max_tokens = args.max_seq_len, 
        layers_p = args.num_layers, 
        num_dev = num_dev,
        ctx = ctx, _fast_init = False) 
    
    if args.freeze_bb == 'ft':
        for (param_name, param) in vae.named_parameters():
            param.requires_grad = True
    elif args.freeze_bb == 'freeze':
        for (param_name, param) in vae.named_parameters():
            if re.fullmatch(".*vae.*", param_name):
                param.requires_grad = True
                # print(f"Trainable: {param_name}")
            else:
                param.requires_grad = False
    
    return vae, tokenizer, config  


if __name__ == "__main__": 
    model, token = get_latent_vae_tokenizer(nullcontext())

    x = torch.arange(100).view(2, 50)
    # model.bart_autoencode(x, torch.zeros_like(x), 1)
