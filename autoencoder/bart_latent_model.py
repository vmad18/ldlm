from contextlib import nullcontext
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from einops import rearrange, repeat

from transformers import AutoTokenizer, PreTrainedTokenizerBase, PretrainedConfig
from transformers.models.bart.modeling_bart import BartForConditionalGeneration 

from ae import AutoEncoder, Config 
from contextlib import nullcontext
from typing import Tuple 

import re

class LatentEncoderConfig(Config): 
    
    dim: int = 768
    latent_dim: int = 768 * 2

    num_latents: int = 32 
    dim_head: int = 128
    
    max_tokens: int = 64 # 1024

    expansion_factor: int = 4

    use_rope: bool = True
    base: int = int(1e5)
    qk_norm: bool = False

    layers_p = 5
    dev = "cuda"


class LatentDecoderConfig(Config): 
    
    dim: int = 2 * 768
    latent_dim: int = 768

    num_latents: int = 64 # 1024 
    dim_head: int = 128
    
    max_tokens: int = 32

    expansion_factor: int = 4

    use_rope: bool = True
    base: int = int(1e5)
    qk_norm: bool = False

    layers_p = 5
    dev = "cuda"


class LatentAEModel(BartForConditionalGeneration): 

    def __init__(self, 
                 config, 
                 ctx = nullcontext(), 
                 num_dev: int = 1) -> None: 
        super().__init__(config) 

        self.ae = AutoEncoder(LatentEncoderConfig(), LatentDecoderConfig())
        self.num_dev = num_dev
        self.freeze = ctx

    def get_bart_encodings(self, input_ids: torch.Tensor, attn_mask: torch.Tensor):
        with self.freeze:
            if self.num_dev > 1: 
                encoder_outs = self.module.get_encoder()(input_ids = input_ids, attention_mask = attn_mask.bool())
            else: 
                encoder_outs = self.get_encoder()(input_ids = input_ids, attention_mask = attn_mask.bool())
        return encoder_outs

    def get_latents(self, input_ids: torch.Tensor, attn_mask) -> torch.Tensor: 
        bart_encodings = self.get_bart_encodings(input_ids, attn_mask)
        return self.ae.encode(bart_encodings, attn_mask.bool()) 

    # decode diffused latent 
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.ae.decode(latent)

    def bart_autoencode(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        *_, s = input_ids.shape 
        bart_encodings = self.get_bart_encodings(input_ids, attn_mask)
        bart_encodings["last_hidden_state"] = self.ae(bart_encodings[0], attn_mask.bool())[..., :s, :]
        return bart_encodings


def get_latent_ae_tokenizer(args, ctx, num_dev: int = 1) -> Tuple[LatentAEModel, PreTrainedTokenizerBase, PretrainedConfig]:
    config = BartForConditionalGeneration.from_pretrained("facebook/bart-base").config
    ae = LatentAEModel.from_pretrained("facebook/bart-base", config=config, ctx=ctx, num_dev = num_dev, _fast_init=False) 
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained("facebook/bart-base") 

    if args.freeze_bb == 'ft':
        for (param_name, param) in lm.named_parameters():
            param.requires_grad = True
    elif args.freeze_bb == 'freeze':
        for (param_name, param) in ae.named_parameters():
            if re.fullmatch(".*ae.*", param_name):
                param.requires_grad = True
                print(f"Trainable: {param_name}")
            else:
                param.requires_grad = False
    
    return ae, tokenizer, config


if __name__ == "__main__": 
    model, token, cfg = get_latent_ae_tokenizer(nullcontext())

    x = torch.arange(100).view(2, 50)
    model.bart_autoencode(x, torch.zeros_like(x), 1)
