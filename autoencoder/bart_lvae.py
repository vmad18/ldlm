from contextlib import nullcontext
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import math

from einops import rearrange, repeat

from transformers import AutoTokenizer, PreTrainedTokenizerBase, PretrainedConfig, T5Config, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput

from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers import AutoTokenizer, PreTrainedTokenizerBase, PretrainedConfig, BartConfig
from transformers.models.bart.modeling_bart import BartForConditionalGeneration 


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


class LatentVAEModel(BartForConditionalGeneration):

    def __init__(self,
                 config: BartConfig,
                 # All other parameters are now passed to the `from_bart_pretrained` classmethod
                 ) -> None:
        # We override the init completely. The Bart parts will be initialized
        # by calling BartForConditionalGeneration.from_pretrained(...) before this.
        # This __init__ is now only responsible for initializing the VAE parts,
        # which will be attached to the BART model instance.
        super().__init__(config)

        # The VAE components will be added dynamically.
        # We still need to define them here for type-hinting and IDE support.
        self.vae_encoder: Optional[nn.Module] = None

        self.proj_in: Optional[nn.Module] = None
        self.proj_out: Optional[nn.Module] = None

        self.vae_post_layernorm: Optional[nn.LayerNorm] = None
        self.latent_dim: Optional[int] = None
        self.num_latents: Optional[int] = None
        self.use_precomputed_latents: bool = False

    @classmethod
    def from_bart_pretrained(
        cls,
        bart_model: BartForConditionalGeneration,
        dim: int,
        latent_dim: int,
        num_latents: int,
        use_precomputed_latents: bool = False,
        create_encoder: bool = True,
        **kwargs # This will contain expansion_factor, dim_head, etc.
    ):
        # This classmethod takes a fully initialized BART model and attaches
        # our custom VAE components to it.
        
        # We can treat the passed bart_model as an instance of our class.
        model = bart_model
        model.__class__ = cls # "Typecast" the instance to our class

        # Store VAE config
        model.latent_dim = latent_dim
        model.num_latents = num_latents
        model.use_precomputed_latents = use_precomputed_latents

        # Initialize VAE components
        bart_dim = model.config.d_model
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
        model.proj_in = nn.Linear(bart_dim, vae_model_dim)
        model.proj_out = nn.Linear(vae_model_dim, bart_dim)

        return model

    def get_bart_encodings(self, 
                           input_ids, 
                           attn_mask=None):

        # if self.encoder is None:
        #     raise ValueError("Cannot get BART encodings when the model is in pre-computed latent mode (encoder is deleted).")
        
        with torch.no_grad():
            # Ensure attention mask is boolean
            if attn_mask is not None and attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.bool()

            # The base BART's encoder returns a BaseModelOutput object
            encoder_outputs = self.get_encoder()(input_ids=input_ids, attention_mask=attn_mask)
        return encoder_outputs # This is an object with a .last_hidden_state attribute

    def latents_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Takes BART embeddings and returns VAE latents (mu).
        This is for the diffusion training pipeline with precomputed BART embeddings.
        """
        embeddings = embeddings.to(self.dtype)
        projected_embeddings = self.proj_in(embeddings)
        moments = self.vae.encode(projected_embeddings)
        return self.vae.reparameterize(*moments, only_mu=True)

    @torch.no_grad()
    def decode_latent(self, latents, max_length=512, **kwargs):
        """ Decodes latents from the diffusion model back into text. """
        
        # --- UPDATED: Use new proj_up path ---
        decoded_from_vae = self.vae.decode(latents)
        projected_output = self.proj_out(decoded_from_vae)

        encoder_outputs = BaseModelOutput(last_hidden_state=projected_output)

        gen_kwargs = {
            "max_length": 512,
            "do_sample": True,
            "top_p": 0.95,
            "num_beams": 1,
            **kwargs
        }

        return self.generate(
            encoder_outputs=encoder_outputs,
            **gen_kwargs
        )

    def autoencode(self, 
                   input_ids: torch.Tensor, 
                   attention_mask: Optional[torch.Tensor] = None) -> Tuple[Any, torch.Tensor, torch.Tensor]:
        # Get BART's raw embeddings
        bart_encoder_outputs = self.get_bart_encodings(input_ids, attn_mask=attention_mask)
        bart_embeddings = bart_encoder_outputs.last_hidden_state

        # --- UPDATED: Use new proj_down/proj_up path ---
        projected_embeddings = self.proj_in(bart_embeddings)
        recon_embeddings, mu, log_var = self.vae(projected_embeddings)
        # Project back up to BART dimension for the decoder
        final_recon_embeddings = self.proj_out(recon_embeddings)
        
        # Calculate VAE loss in the projected space
        vae_loss_dict = self.vae.cont_loss_func(final_recon_embeddings, bart_embeddings, mu, log_var)
        

        # Create a new encoder_outputs object for the BART decoder
        final_encoder_outputs = BaseModelOutput(
            last_hidden_state=final_recon_embeddings,
            # Make sure to carry over other attributes if needed, e.g., attentions
            attentions=bart_encoder_outputs.attentions
        )
        
        return final_encoder_outputs, vae_loss_dict, mu

    def encode(self, precomputed_latents: torch.Tensor) -> Tuple[Any, dict]:
        """
        Takes pre-computed BART latents, runs them through the VAE, and returns the VAE's output and loss.
        """
        # --- UPDATED: Use new proj_down path ---
        projected_latents = self.proj_in(precomputed_latents)
        recon_latents, mu, log_var = self.vae(projected_latents)
        # Loss is calculated in the projected space
        vae_loss_dict = self.vae.cont_loss_func(recon_latents, projected_latents, mu, log_var)
        return recon_latents, vae_loss_dict['kld_loss']

    def decode_loss(self, original_latents: torch.Tensor, vae_output: Any) -> torch.Tensor:
        """
        Computes the reconstruction loss between original BART latents and the VAE's output.
        The loss is now computed in the VAE's projected space.
        """
        return F.mse_loss(vae_output, self.proj_out(original_latents)) # oh... why do we do this in the VAE's latent space, comp less expensive?

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
            # This path is used by .generate() for full BART only computation
            # Make sure to pass along any other kwargs .generate() might be using
            return super().forward(labels=labels, encoder_outputs=encoder_outputs, **kwargs)

        if input_latents is not None:
            # Pre-computed latent path (training the VAE on saved BART embeddings)

            # Cast input latents to the model's float dtype to prevent mismatch
            input_latents = input_latents.to(self.dtype)
            if not self.use_precomputed_latents:
                raise ValueError("`input_latents` was provided, but the model was not initialized with `use_precomputed_latents=True`.")
            
            # --- UPDATED: Use new projection paths and loss calculation ---
            projected_embeddings = self.proj_in(input_latents)
            recon_embeddings, mu, log_var = self.vae(projected_embeddings)

            # Calculate VAE loss in the projected space
            vae_loss_dict = self.vae.cont_loss_func(recon_embeddings, projected_embeddings, mu, log_var)
            recon_loss = vae_loss_dict['reconstruction_loss']
            kld_loss = vae_loss_dict['kld_loss']
            
            # For generation, project back up and wrap in BaseModelOutput
            reconstructed_bart_embeddings = self.proj_out(recon_embeddings)
            enc_outs = BaseModelOutput(last_hidden_state=reconstructed_bart_embeddings)
            
            
            # lm_loss = super().forward(labels=labels, encoder_outputs=enc_outs, **kwargs).loss

            return {
                # 'lm_loss': lm_loss,
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

            # autoencode handles the full BART->VAE->BART path
            encoder_outputs_from_vae, vae_loss, _ = self.autoencode(input_ids, attention_mask=attention_mask)
            
            # Pass the reconstructed embeddings to the BART decoder to calculate LM loss
            lm_loss = super().forward(labels=labels, encoder_outputs=encoder_outputs_from_vae, **kwargs).loss

            return {
                'lm_loss': lm_loss,
                'vae_loss': vae_loss,
                'encoder_outputs': encoder_outputs_from_vae
            }

        raise ValueError("Either `input_ids` or `input_latents` must be provided.")


def get_latent_vae_tokenizer_bart(
    args,
    ctx = nullcontext(),
    num_dev: int = 1,
    base_bart: str = "facebook/bart-base",
    create_encoder: bool = False,
) -> Tuple[LatentVAEModel, PreTrainedTokenizerBase, PretrainedConfig]:

    """
    Returns a BART model with an attached VAE, its tokenizer, and the original BART config.
    The VAE components are initialized and attached inside this function.
    """
    
    # Load the base BART model and tokenizer
    with ctx:
        bart_model = BartForConditionalGeneration.from_pretrained(base_bart)
        tokenizer = AutoTokenizer.from_pretrained(base_bart, use_fast=False)

    # These are the args that are defined in `ae.Config`
    vae_params = {
        'expansion_factor': getattr(args.model_config, 'expansion_factor', 4),
        'dim_head': getattr(args.model_config, 'dim_head', 128),
        # The VAE config uses 'num_layers', but the Perceiver code expects 'layers_p'
        'layers_p': getattr(args.model_config, 'num_layers', 8), 
        'use_rope': getattr(args.model_config, 'use_rope', True),
        'qk_norm': getattr(args.model_config, 'qk_norm', False),
        'max_tokens': getattr(args.model_config, 'max_seq_len', 1024),
        # 'model_dim' is equivalent to 'dim' for the VAE's internal workings.
        # We pass it explicitly below to avoid confusion.
        # 'latent_dim' and 'num_latents' are also passed explicitly.
    }


    print("================================== VAE PARAMS ===================================")
    print(vae_params)
    print("=================================================================================")
    
    # Attach our VAE and custom methods to the BART model instance
    model = LatentVAEModel.from_bart_pretrained(
        bart_model,
        # VAE structural parameters
        dim = args.model_config.d_model,
        latent_dim = args.model_config.latent_dim,
        num_latents = args.model_config.num_latents,
        # Control flow parameters
        use_precomputed_latents = getattr(args, 'use_precomputed_latents', False),
        create_encoder = getattr(args, 'create_encoder', True),
        # Pass the rest of the VAE config
        **vae_params
    )


    cfg = bart_model.config
    del bart_model
    if not args.freeze_bb:
        for (param_name, param) in model.named_parameters():
            param.requires_grad = True
    elif args.freeze_bb:
        for (param_name, param) in model.named_parameters():
            if re.fullmatch(".*vae.*", param_name):
                param.requires_grad = True
            else:
                param.requires_grad = False
        
    return model, tokenizer, cfg


if __name__ == "__main__": 
    model, token = get_latent_vae_tokenizer(nullcontext())

    x = torch.arange(100).view(2, 50)
    # model.bart_autoencode(x, torch.zeros_like(x), 1)
