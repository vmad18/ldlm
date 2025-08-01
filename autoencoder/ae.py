import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import sys

from einops import rearrange, repeat
from math import sqrt, log  

from typing import Tuple, Optional

class Config:

    def __init__(self,
            dim: int = 1024,
            num_latents: int = 16,
            latent_dim: int = 1024,
            model_dim: int = 1024,
            dim_head: int = 128,
            max_tokens: int = 1024,
            expansion_factor: int = 4,
            use_rope: bool = True,
            base: int = int(1e5),
            qk_norm: bool = False,
            layers_p = 8,
            dev = "cuda" if torch.cuda.is_available() else "cpu",) -> None:

        self.dim: int = dim
        self.num_latents: int = num_latents
        self.latent_dim: int = latent_dim
        self.model_dim: int = model_dim
        self.dim_head: int = dim_head
        self.max_tokens: int = max_tokens

        self.expansion_factor: int = expansion_factor

        self.use_rope: bool = use_rope
        self.base: int = base
        self.qk_norm: bool = qk_norm

        self.layers_p = layers_p
        self.dev = dev


def create_enc_dec_cfg(
    **kwargs
) -> Tuple[Config, Config]:
    # Encoder config: Takes in a sequence of `max_tokens` vectors, each of dimension `dim`.
    # Outputs `num_latents` vectors, each of dimension `latent_dim`.
    enc_cfg = Config(**kwargs)

    # For the decoder, we swap some params. Pop `max_tokens` from a copy of kwargs
    # to avoid passing it twice to the Config constructor.
    dec_kwargs = kwargs.copy()
    dec_kwargs['latent_dim'], dec_kwargs['dim'] = dec_kwargs['dim'], dec_kwargs['latent_dim']
    dec_kwargs['num_latents'], dec_kwargs['max_tokens'] = dec_kwargs['max_tokens'], dec_kwargs['num_latents']

    # Decoder config: Takes in `num_latents` vectors (from encoder), each of dimension `latent_dim`.
    # Outputs `max_tokens` vectors (reconstructed), each of dimension `dim`.
    dec_cfg = Config(**dec_kwargs)
    return enc_cfg, dec_cfg


class RoPE(nn.Module):

    def __init__(self, cfg: Config, dim: int, scaling: float = 1., device: str = "cuda" if torch.cuda.is_available() else "cpu") -> None:
        super().__init__()

        self.dim = dim 
        self.base = cfg.base 
        self.scaling = scaling

        self.max_toks = cfg.max_tokens + cfg.num_latents
        
        self.dev = device

    def comp_rots(self) -> torch.Tensor: 
        theta = torch.exp(-log(self.base) * torch.arange(0, self.dim, 2, device=self.dev, dtype=torch.int64) / self.dim)[None, ...]

        m = torch.arange(self.max_toks, device = self.dev, dtype=torch.float32)[..., None].float() 
        freqs = m * theta 
        mag = torch.ones_like(freqs, device = self.dev)
        
        return torch.polar(mag, freqs)
    
    def _to_complex(self, x: torch.Tensor) -> torch.Tensor:
        return torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))

    def _to_real(self, x: torch.Tensor) -> torch.Tensor: 
        return torch.view_as_real(x)

    def forward(self, q: torch.Tensor, k: torch.Tensor, shift: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        *_, s1, d = q.shape 
        *_, s2, d = k.shape 

        dtype = q.dtype


        q, k = q.float(), k.float() 
        rots_q = self.comp_rots()[shift : shift + s1].reshape(1, 1, s1, d // 2)
        rots_k = self.comp_rots()[shift : shift + s2].reshape(1, 1, s2, d // 2)

        _q = self._to_complex(q) * rots_q
        _k = self._to_complex(k) * rots_k

        rq = self._to_real(_q).to(dtype)
        rk = self._to_real(_k).to(dtype) 

        return rq.reshape(*rq.shape[:-2], d), rk.reshape(*rk.shape[:-2], d)

class AbsolutePositionalEmbedding(nn.Module): 

    def __init__(self, cfg: Config, dim: int):
        super().__init__() 

        self.scale = 1. / sqrt(dim)
        self.embed = nn.Embedding(cfg.max_tokens, dim, device=cfg.dev)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        s = x.shape[-2] 

        pos = torch.arange(s, device=x.device) 
        return self.embed(pos) * self.scale 

class FeedForward(nn.Module):

    def __init__(self, cfg: Config, dim: int) -> None:
        super().__init__()
        self.proj_up = nn.Linear(dim, cfg.expansion_factor * dim, device = cfg.dev)
        
        in_features = cfg.expansion_factor * dim
        out_features = dim

        self.proj_down = nn.Linear(in_features, out_features, device = cfg.dev)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.proj_up(x) 
        x = F.relu(x).square()
        x = self.proj_down(x)
        return x


class PerceiverAttention(nn.Module): 

    def __init__(self, cfg: Config) -> None:
        super().__init__()
       
        inner_dim = cfg.model_dim
        
        self.scale = 1./sqrt(inner_dim)

        self.heads = inner_dim // cfg.dim_head

        self.pre_norm = nn.LayerNorm(cfg.dim, device=cfg.dev)
        self.latent_norm = nn.LayerNorm(cfg.latent_dim, device=cfg.dev) 

        self.proj_q_latent = nn.Linear(cfg.latent_dim, inner_dim, device=cfg.dev) 
        
        self.proj_kv = nn.Linear(cfg.dim, 2 * inner_dim, device=cfg.dev)
        self.proj_kv_latent = nn.Linear(cfg.latent_dim, 2 * inner_dim, device=cfg.dev) 

        if cfg.use_rope:
            self.rope = RoPE(cfg, cfg.dim_head, device=cfg.dev) 
        else:
            self.rope = None

        self.proj_o = nn.Linear(inner_dim, cfg.latent_dim, device=cfg.dev)

    def forward(self, x: torch.Tensor, latents: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = self.pre_norm(x) 
        latents = self.latent_norm(latents) 

        q_lat = self.proj_q_latent(latents) 
        kv_xl = torch.cat([self.proj_kv(x), self.proj_kv_latent(latents)], dim = -2)

        k, v = rearrange(kv_xl, "b s (l d) -> l b s d", l = 2)
        q, k, v = map(lambda t: rearrange(t, "b s (h d) -> b h s d", h = self.heads), (q_lat, k, v))
       
        if self.rope is not None:
            q, k = self.rope(q, k)

        sim = (q @ k.transpose(-1, -2)) * self.scale

        if mask is not None: 
            mask = F.pad(mask, (0, latents.shape[-2]), value = True)
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, torch.finfo(sim.dtype).min)

        attn_score = F.softmax(sim, dim=-1, dtype=torch.float32).to(sim.dtype) 
        out = rearrange((attn_score @ v), "b h s d -> b s (h d)", h = self.heads)
        return self.proj_o(out)


class AutoEncodingBlock(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__() 

        self.attn = PerceiverAttention(cfg) 
        self.ffn1 = FeedForward(cfg, cfg.dim)
        self.ffn2 = FeedForward(cfg, cfg.latent_dim)

        self.ln1 = nn.LayerNorm(cfg.dim, device = cfg.dev)
        self.ln2 = nn.LayerNorm(cfg.latent_dim, device = cfg.dev) 

    def forward(self, x: torch.Tensor, latents: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        latents = self.attn(x, latents, mask) + latents
        x_trans = self.ffn1(self.ln1(x)) + x 
        latents = self.ffn2(self.ln2(latents)) + latents 

        return x_trans, latents

class PerceiverResampler(nn.Module): 

    def __init__(self, cfg: Config) -> None:
        super().__init__()

        self.pos_embed = AbsolutePositionalEmbedding(cfg, cfg.dim)

        self.latents = nn.Parameter(torch.randn((cfg.num_latents, cfg.latent_dim), device=cfg.dev))
        nn.init.normal_(self.latents, std = 0.02) 

        self.blocks = nn.ModuleList([])

        for _ in range(cfg.layers_p):
            self.blocks.append(AutoEncodingBlock(cfg))

        self.f_attn = PerceiverAttention(cfg)
        self.latent_norm = nn.LayerNorm(cfg.latent_dim, device=cfg.dev)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, *_ = x.shape

        x = x + self.pos_embed(x)
        
        latents = repeat(self.latents, "n d -> b n d", b = x.shape[0])

        for i, block in enumerate(self.blocks):
            x, latents = block(x, latents, mask)

        latents = self.f_attn(x, latents, mask)

        latents = self.latent_norm(latents)

        return latents


class VariationalAutoEncoder(nn.Module):
    def __init__(self, cfg_enc: Config, cfg_dec: Config, create_encoder: bool = True) -> None:
        super().__init__()
        if create_encoder:
            self.encoder = PerceiverResampler(cfg_enc)
            self.mu_lsigma = nn.Linear(cfg_enc.latent_dim, 2 * cfg_enc.latent_dim, device=cfg_enc.dev)
        else:
            self.encoder = None
            self.mu_lsigma = None

        self.decoder = PerceiverResampler(cfg_dec)

    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes the input and returns the mean and log-variance vectors."""
        if self.encoder is None:
            raise ValueError("Cannot encode without an encoder. Model was initialized with create_encoder=False.")
        
        enc_output = self.encoder(x, mask)
        mu, log_var = self.mu_lsigma(enc_output).chunk(2, dim=-1)
        
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor, mu_only: bool = False) -> torch.Tensor:
        """Reparameterizes the latent space. If only_mu is True, returns the mean."""
        if mu_only:
            return mu
        std = torch.exp(0.5 * log_var)  
        eps = torch.randn_like(std)     
        return mu + eps * std           # compute the latent vector z

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decodes latents back into the embedding space."""
        return self.decoder(latents)

    def discrete_loss_func(self, recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> dict:
        recon_loss = F.cross_entropy(recon_x, x) # F.mse_loss(recon_x, x, reduction='sum')

        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()

        total_loss = recon_loss + kld_loss
        
        return {'total_loss': total_loss, 'reconstruction_loss': recon_loss, 'kld_loss': kld_loss}

    def cont_loss_func(self, recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> dict:
        if recon_x.shape != x.shape:
            print(f"[VAE ERROR] Shape mismatch in loss calculation:")
            print(f"  --> recon_x shape: {recon_x.shape}")
            print(f"  --> x shape:       {x.shape}")
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kld_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        total_loss = recon_loss + kld_loss
        
        return {'total_loss': total_loss, 'reconstruction_loss': recon_loss, 'kld_loss': kld_loss}
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, mu_only: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x, mask)
        z = self.reparameterize(mu, log_var, mu_only)
        recon_x = self.decode(z)
        return recon_x, mu, log_var


if __name__ == "__main__":
    e_cfg, d_cfg = create_enc_dec_cfg(dim=1024, latent_dim=512, num_latents=16, model_dim=1024, max_tokens=1024)
    attn = VariationalAutoEncoder(e_cfg, d_cfg)
    # This block is for testing purposes.
    # x = torch.randn((3, 1024, 1024)).cuda()
    # recon_x, mu, log_var = attn(x)
    # print(recon_x.shape)
    
    print("VAE initialized successfully for testing.") 
