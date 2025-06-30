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
                 # All other parameters are now passed to the `from_t5_pretrained` classmethod
                 ) -> None:
        # We override the init completely. The T5 parts will be initialized
        # by calling T5ForConditionalGeneration.from_pretrained(...) before this.
        # This __init__ is now only responsible for initializing the VAE parts,
        # which will be attached to the T5 model instance.
        super().__init__(config)

        # The VAE components will be added dynamically.
        # We still need to define them here for type-hinting and IDE support.
        self.vae_encoder: Optional[nn.Module] = None
        self.vae_post_layernorm: Optional[nn.LayerNorm] = None
        self.latent_dim: Optional[int] = None
        self.num_latents: Optional[int] = None
        self.use_precomputed_latents: bool = False

    @classmethod
    def from_t5_pretrained(
        cls,
        t5_model: T5ForConditionalGeneration,
        dim: int,
        latent_dim: int,
        num_latents: int,
        use_precomputed_latents: bool = False,
        create_encoder: bool = True,
        **kwargs # This will contain expansion_factor, dim_head, etc.
    ):
        # This classmethod takes a fully initialized T5 model and attaches
        # our custom VAE components to it.
        
        # We can treat the passed t5_model as an instance of our class.
        model = t5_model
        model.__class__ = cls # "Typecast" the instance to our class

        # Store VAE config
        model.latent_dim = latent_dim
        model.num_latents = num_latents
        model.use_precomputed_latents = use_precomputed_latents

        # Initialize VAE components
        t5_dim = model.config.d_model
        # The VAE's working dimension is now passed explicitly as `dim`.
        vae_model_dim = dim
                
        # The VAE configs are now created correctly.
        kwargs.pop('model_dim', None)
        
        enc_cfg, dec_cfg = create_enc_dec_cfg(
            dim=vae_model_dim,      # Input dimension for the encoder
            latent_dim=latent_dim,     # This is the dimension of the Perceiver's internal latents
            num_latents=num_latents,   # Number of Perceiver internal latents
            **kwargs
        )
        
        model.vae = VariationalAutoEncoder(
            cfg_enc=enc_cfg,
            cfg_dec=dec_cfg,
            create_encoder=create_encoder,
        )

        # --- NEW: Attach projection layers directly to the vae module ---
        model.vae.proj_down = nn.Linear(t5_dim, vae_model_dim)
        model.vae.proj_up = nn.Linear(vae_model_dim, t5_dim)

        return model

    def get_t5_encodings(self, input_ids, attn_mask=None):
        if self.encoder is None:
            raise ValueError("Cannot get T5 encodings when the model is in pre-computed latent mode (encoder is deleted).")
        
        # Ensure attention mask is boolean
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.bool()

        # The base T5Model's encoder returns a BaseModelOutput object
        encoder_outputs = self.get_encoder()(input_ids=input_ids, attention_mask=attn_mask)
        return encoder_outputs # This is an object with a .last_hidden_state attribute

    def latents_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Takes T5 embeddings and returns VAE latents (mu).
        This is for the diffusion training pipeline with precomputed T5 embeddings.
        """
        embeddings = embeddings.to(self.dtype)
        projected_embeddings = self.vae.proj_down(embeddings)
        moments = self.vae.encode(projected_embeddings)
        return self.vae.reparameterize(*moments, only_mu=True)

    @torch.no_grad()
    def decode_latent(self, latents, max_length=512, **kwargs):
        """
        Decodes latents from the diffusion model back into text.
        """
        # --- UPDATED: Use new proj_up path ---
        decoded_from_vae = self.vae.decode(latents)
        projected_output = self.vae.proj_up(decoded_from_vae)

        encoder_outputs = BaseModelOutput(last_hidden_state=projected_output)

        gen_kwargs = {
            "max_length": max_length,
            "do_sample": True,
            "top_p": 0.95,
            "num_beams": 1,
            **kwargs
        }

        return self.generate(
            encoder_outputs=encoder_outputs,
            **gen_kwargs
        )

    def autoencode(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[Any, torch.Tensor, torch.Tensor]:
        # Get T5's raw embeddings
        t5_encoder_outputs = self.get_t5_encodings(input_ids, attn_mask=attention_mask)
        t5_embeddings = t5_encoder_outputs.last_hidden_state

        # --- UPDATED: Use new proj_down/proj_up path ---
        projected_embeddings = self.vae.proj_down(t5_embeddings)
        recon_embeddings, mu, log_var = self.vae(projected_embeddings)
        
        # Calculate VAE loss in the projected space
        vae_loss_dict = self.vae.cont_loss_func(recon_embeddings, projected_embeddings, mu, log_var)
        
        # Project back up to T5 dimension for the decoder
        final_recon_embeddings = self.vae.proj_up(recon_embeddings)
        
        # Create a new encoder_outputs object for the T5 decoder
        final_encoder_outputs = BaseModelOutput(
            last_hidden_state=final_recon_embeddings,
            # Make sure to carry over other attributes if needed, e.g., attentions
            attentions=t5_encoder_outputs.attentions
        )
        
        return final_encoder_outputs, vae_loss_dict['total_loss'], mu

    def encode(self, precomputed_latents: torch.Tensor) -> Tuple[Any, dict]:
        """
        Takes pre-computed T5 latents, runs them through the VAE, and returns the VAE's output and loss.
        """
        # --- UPDATED: Use new proj_down path ---
        projected_latents = self.vae.proj_down(precomputed_latents)
        recon_latents, mu, log_var = self.vae(projected_latents)
        # Loss is calculated in the projected space
        vae_loss_dict = self.vae.cont_loss_func(recon_latents, projected_latents, mu, log_var)
        return recon_latents, vae_loss_dict['kld_loss']

    def decode_loss(self, original_latents: torch.Tensor, vae_output: Any) -> torch.Tensor:
        """
        Computes the reconstruction loss between original T5 latents and the VAE's output.
        The loss is now computed in the VAE's projected space.
        """
        return F.mse_loss(vae_output, self.vae.proj_down(original_latents))

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
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
            # Pre-computed latent path (training the VAE on saved T5 embeddings)

            # Cast input latents to the model's float dtype to prevent mismatch
            input_latents = input_latents.to(self.dtype)

            if not self.use_precomputed_latents:
                raise ValueError("`input_latents` was provided, but the model was not initialized with `use_precomputed_latents=True`.")
            
            # --- UPDATED: Use new projection paths and loss calculation ---
            projected_embeddings = self.vae.proj_down(input_latents)
            recon_embeddings, mu, log_var = self.vae(projected_embeddings)

            # Calculate VAE loss in the projected space
            vae_loss_dict = self.vae.cont_loss_func(recon_embeddings, projected_embeddings, mu, log_var)
            recon_loss = vae_loss_dict['reconstruction_loss']
            kld_loss = vae_loss_dict['kld_loss']
            
            # For generation, project back up and wrap in BaseModelOutput
            reconstructed_t5_embeddings = self.vae.proj_up(recon_embeddings)
            enc_outs = BaseModelOutput(last_hidden_state=reconstructed_t5_embeddings)
            
            return {
                'reconstruction_loss': recon_loss,
                'kld_loss': kld_loss,
                'encoder_outputs': enc_outs
            }

        if input_ids is not None:
            # On-the-fly tokenization path (end-to-end training)
            if self.use_precomputed_latents:
                raise ValueError("`input_ids` was provided, but the model was initialized with `use_precomputed_latents=True`.")
            
            # Get attention mask to pass to autoencode
            attention_mask = kwargs.get('attention_mask', None)

            # autoencode handles the full T5->VAE->T5 path
            encoder_outputs_from_vae, vae_loss, _ = self.autoencode(input_ids, attention_mask=attention_mask)
            
            # Pass the reconstructed embeddings to the T5 decoder to calculate LM loss
            lm_loss = super().forward(labels=labels, encoder_outputs=encoder_outputs_from_vae, **kwargs).loss

            return {
                'lm_loss': lm_loss,
                'vae_loss': vae_loss,
                'encoder_outputs': encoder_outputs_from_vae
            }

        raise ValueError("Either `input_ids` or `input_latents` must be provided.")


def get_latent_vae_tokenizer_t5(
    args,
    ctx=nullcontext(),
    num_dev: int = 1,
    base_t5: str = "bigscience/T0pp",
    create_encoder: bool = False,
) -> Tuple[LatentVAEModel, PreTrainedTokenizerBase, PretrainedConfig]:
    """
    Returns a T5 model with an attached VAE, its tokenizer, and the original T5 config.
    The VAE components are initialized and attached inside this function.
    """
    
    # Load the base T5 model and tokenizer
    with ctx:
        t5_model = T5ForConditionalGeneration.from_pretrained(base_t5)
        tokenizer = AutoTokenizer.from_pretrained(base_t5, use_fast=False)

    # These are the args that are defined in `ae.Config`
    vae_params = {
        'expansion_factor': getattr(args, 'expansion_factor', 4),
        'dim_head': getattr(args, 'dim_head', 128),
        # The VAE config uses 'num_layers', but the Perceiver code expects 'layers_p'
        'layers_p': getattr(args, 'num_layers', 8), 
        'use_rope': getattr(args, 'use_rope', True),
        'qk_norm': getattr(args, 'qk_norm', False),
        'max_tokens': getattr(args, 'max_seq_len', 1024),
        # 'model_dim' is equivalent to 'dim' for the VAE's internal workings.
        # We pass it explicitly below to avoid confusion.
        # 'latent_dim' and 'num_latents' are also passed explicitly.
    }

    # Attach our VAE and custom methods to the T5 model instance
    model = LatentVAEModel.from_t5_pretrained(
        t5_model,
        # VAE structural parameters
        dim=args.d_model,
        latent_dim=args.latent_dim,
        num_latents=args.num_latents,
        # Control flow parameters
        use_precomputed_latents=getattr(args, 'use_precomputed_latents', False),
        create_encoder=getattr(args, 'create_encoder', True),
        # Pass the rest of the VAE config
        **vae_params
    )

    # if create_encoder:
    #     print("T5 encoder is present.")
    # else:
    #     # For pre-computed latent training, we don't need the T5 encoder.
    #     # Deleting it saves a significant amount of memory.
    #     del model.encoder
    #     model.encoder = None # Set to None for clarity
    #     print("T5 encoder has been deleted to save memory.")


    cfg = t5_model.config
    del t5_model
    if args.freeze_bb == 'ft':
        for (param_name, param) in model.named_parameters():
            param.requires_grad = True
    elif args.freeze_bb == 'freeze':
        for (param_name, param) in model.named_parameters():
            if re.fullmatch(".*vae.*", param_name):
                param.requires_grad = True
                # print(f"Trainable: {param_name}")
            else:
                param.requires_grad = False
        
    return model, tokenizer, cfg


if __name__ == "__main__": 
    model, token = get_latent_vae_tokenizer(nullcontext())

    x = torch.arange(100).view(2, 50)
    # model.bart_autoencode(x, torch.zeros_like(x), 1)
