from contextlib import nullcontext
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from einops import rearrange, repeat

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .ae import VariationalAutoEncoder, Config, create_enc_dec_cfg
from contextlib import nullcontext
from typing import Tuple, Optional

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

class LatentVAEModel(nn.Module): 

    def __init__(self, 
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
                 dev: str = "cuda" if torch.cuda.is_available() else "cpu") -> None: 
        super().__init__()

        cfg_enc, cfg_dec = create_enc_dec_cfg(dim=d_model, 
                                              latent_dim=latent_dim, 
                                              num_latents=num_latents, 
                                              dim_head=dim_head,
                                              max_tokens=max_tokens, 
                                              expansion_factor=expansion_factor, use_rope=use_rope, 
                                              base=rope_base, qk_norm=qk_norm, layers_p=layers_p, dev=dev)

        self.vae = VariationalAutoEncoder(cfg_enc, cfg_dec)

        self.embed = nn.Embedding(vocab_size, d_model, device=dev) 
        self.dembed_head = nn.Linear(d_model, vocab_size, device=dev)
        
        self.vocab_size = vocab_size
        self.dim = cfg_enc.dim
        self.latent_dim = cfg_enc.latent_dim
        self.num_latents = cfg_enc.num_latents

        self.max_tokens = cfg_enc.max_tokens
        self.freeze = ctx

    def get_latents(self, input_ids: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor: 
        x = self.embed(input_ids) 
        if attn_mask is None: 
            attn_mask = torch.ones_like(input_ids)
        return self.vae.reparameterize(*self.vae.encode(x, attn_mask.bool()), only_mu=True)

    # decode diffused latent 
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        decoded_embeds = self.vae.decode(latent)
        return self.dembed_head(decoded_embeds)

    def autoencode(self, input_ids: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> dict:
        *_, s = input_ids.shape 
        embeddings = self.embed(input_ids) 
        
        if attn_mask is None: 
            attn_mask = torch.ones_like(input_ids)

        recon_encs, mu, log_var = self.vae(embeddings, attn_mask.bool())
        recon_encs = self.dembed_head(recon_encs[..., :s, :])
        return self.vae.discrete_loss_func(recon_encs.view(-1, self.vocab_size), input_ids.view(-1), mu, log_var)

    def forward(self, input_ids: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> dict:
        return self.autoencode(input_ids, attn_mask)

    def compile_transformer_blocks(self, compile_kwargs: dict):
        """
        Compiles the transformer blocks within the VAE's encoder and decoder.
        This is a form of partial compilation to accelerate the most intensive parts of the model.
        """
        for block in self.vae.encoder.layers:
            block.compile(**compile_kwargs)
        
        for block in self.vae.decoder.layers:
            block.compile(**compile_kwargs)


def get_latent_vae_tokenizer(model_cfg, tokenizer_name: str = "meta-llama/Llama-3.2-1B") -> Tuple[LatentVAEModel, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    vae = LatentVAEModel(
        vocab_size = len(tokenizer),
        d_model = model_cfg.d_model, 
        latent_dim = model_cfg.latent_dim, 
        num_latents = model_cfg.num_latents, 
        dim_head = model_cfg.dim_head, 
        max_tokens = model_cfg.max_seq_len, 
        layers_p = model_cfg.num_layers) 
    
    return vae, tokenizer 


if __name__ == "__main__": 
    model, token = get_latent_vae_tokenizer(nullcontext())

    x = torch.arange(100).view(2, 50)
    # model.bart_autoencode(x, torch.zeros_like(x), 1)