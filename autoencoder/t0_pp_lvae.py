from contextlib import nullcontext
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from einops import rearrange, repeat

from transformers import AutoTokenizer, PreTrainedTokenizerBase, PretrainedConfig, T5Config

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
    dev = "cuda"


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
    dev = "cuda"


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
                 dev: str = "cuda",) -> None: 
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
        
        recon_encs, mu, log_var = self.vae(encodgins[0], attn_mask.bool())
        recon_encs = self.proj_up(recon_encs[..., :s, :]) 
        encodings["last_hidden_state"] = recon_encs 
        return encodings, self.vae.cont_loss_func(recon_encs, encodings[0], mu, log_var)

    

    # def forward(self, input_ids: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> dict:
    #     return self.autoencode(input_ids, attn_mask)
    

def get_latent_vae_tokenizer_t5(args, ctx, num_dev: int = 1, base_t5: str = "bigscience/T0pp") -> Tuple[LatentVAEModel, PreTrainedTokenizerBase, PretrainedConfig]:
    config = T5ForConditionalGeneration.from_pretrained(base_t5).config 
    tokenizer = AutoTokenizer.from_pretrained(base_t5)

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
                print(f"Trainable: {param_name}")
            else:
                param.requires_grad = False
    
    return vae, tokenizer, config  


if __name__ == "__main__": 
    model, token = get_latent_vae_tokenizer(nullcontext())

    x = torch.arange(100).view(2, 50)
    # model.bart_autoencode(x, torch.zeros_like(x), 1)
