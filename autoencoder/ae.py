import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from einops import rearrange, repeat
from math import sqrt, log  

from typing import Tuple, Optional

class Config:

    def __init__(self,
            dim: int = 1024,
            num_latents: int = 16,
            latent_dim: int = 1024,
            dim_head: int = 128,
            max_tokens: int = 1024,
            expansion_factor: int = 4,
            use_rope: bool = True,
            base: int = int(1e5),
            qk_norm: bool = False,
            layers_p = 8,
            dev = "cuda",) -> None:

        self.dim: int = dim
        self.num_latents: int = num_latents
        self.latent_dim: int = latent_dim
        self.dim_head: int = dim_head
        self.max_tokens: int = max_tokens

        self.expansion_factor: int = expansion_factor

        self.use_rope: bool = use_rope
        self.base: int = base
        self.qk_norm: bool = qk_norm

        self.layers_p = layers_p
        self.dev = dev

class EncConfig(Config):

    dim: int = 512
    latent_dim: int = 1024
    num_latents: int = 16
    dim_head: int = 128
    max_tokens: int = 1024

    expansion_factor: int = 4

    use_rope: bool = True
    base: int = int(1e5)
    qk_norm: bool = False

    layers_p = 8

    dev = "cuda"

class DecConfig(Config):

    dim: int = 1024
    latent_dim: int = 512
    num_latents: int = 1024
    dim_head: int = 128
    max_tokens: int = 16

    expansion_factor: int = 4

    use_rope: bool = True
    base: int = int(1e5)
    qk_norm: bool = False

    layers_p = 8

    dev = "cuda"


def create_enc_dec_cfg(
    dim: int = 1024,
    latent_dim: int = 512,
    num_latents: int = 1024,
    dim_head: int = 128,
    max_tokens: int = 16,
    expansion_factor: int = 4,
    use_rope: bool = True,
    base: int = int(1e5),
    qk_norm: bool = False,
    layers_p = 8,
    dev = "cuda",
) -> Tuple[Config, Config]:
    enc_cfg = Config(dim=dim,
                     latent_dim=latent_dim,
                     num_latents=num_latents,
                     dim_head=dim_head, max_tokens=max_tokens,
                     expansion_factor=expansion_factor, use_rope=use_rope,
                     base=base, qk_norm=qk_norm, layers_p=layers_p, dev=dev)

    dec_cfg = Config(dim=latent_dim,
                     latent_dim=dim,
                     num_latents=max_tokens,
                     dim_head=dim_head, max_tokens=num_latents,
                     expansion_factor=expansion_factor, use_rope=use_rope,
                     base=base, qk_norm=qk_norm, layers_p=layers_p, dev=dev)
    return enc_cfg, dec_cfg


class RoPE(nn.Module):

    def __init__(self, cfg: Config, dim: int, scaling: float = 1., device ="cuda") -> None:
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

    def __init__(self, cfg: Config):
        super().__init__() 

        self.scale = 1. / sqrt(cfg.dim)
        self.embed = nn.Embedding(cfg.max_tokens, cfg.dim, device=cfg.dev)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        s = x.shape[-2] 

        pos = torch.arange(s, device=x.device) 
        return self.embed(pos) * self.scale 

class FeedForward(nn.Module):

    def __init__(self, cfg: Config, dim: int) -> None:
        super().__init__()
        self.proj_up = nn.Linear(dim, cfg.expansion_factor * dim, device = cfg.dev)
        self.proj_down = nn.Linear(cfg.expansion_factor * dim, dim, device = cfg.dev)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.proj_up(x) 
        x = F.relu(x).square()
        x = self.proj_down(x)
        return x


class PerceiverAttention(nn.Module): 

    def __init__(self, cfg: Config) -> None:
        super().__init__()
       
        inner_dim = max(cfg.dim, cfg.latent_dim)
        
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

        self.pos_embed = AbsolutePositionalEmbedding(cfg)

        self.latents = nn.Parameter(torch.randn((cfg.num_latents, cfg.latent_dim), device=cfg.dev))
        nn.init.normal_(self.latents, std = 0.02) 

        self.layers = nn.ModuleList([])

        for _ in range(cfg.layers_p):
            self.layers.append(AutoEncodingBlock(cfg))

        self.f_attn = PerceiverAttention(cfg)
        self.latent_norm = nn.LayerNorm(cfg.latent_dim, device = cfg.dev)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, *_ = x.shape

        x = x + self.pos_embed(x)
        latents = repeat(self.latents, "s d -> b s d", b = b)
        for layer in self.layers:
            x, latents = layer(x, latents, mask) 
        
        latents = self.f_attn(x, latents, mask)
        return self.latent_norm(latents)


class VariationalAutoEncoder(nn.Module):
    def __init__(self, cfg_enc: Config, cfg_dec: Config) -> None:
        super().__init__()

        self.encoder = PerceiverResampler(cfg_enc)
        self.decoder = PerceiverResampler(cfg_dec)
        
        self.mu_lsigma = nn.Linear(cfg_enc.latent_dim, 2 * cfg_enc.latent_dim, device=cfg_enc.dev)

    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes the input and returns the mean and log-variance vectors."""
        enc_output = self.encoder(x, mask)
        
        mu, log_var = self.mu_lsigma(enc_output).chunk(dim=-1)
        
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor, only_mu: bool = False) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)  
        eps = torch.randn_like(std)     
        if only_mu:
            return mu 
        return mu + eps * std           # compute the latent vector z

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def discrete_loss_func(self, recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> dict:
        recon_loss = F.cross_entropy(recon_x, x) # F.mse_loss(recon_x, x, reduction='sum')

        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        total_loss = recon_loss + kld_loss
        
        return {'total_loss': total_loss, 'reconstruction_loss': recon_loss, 'kld_loss': kld_loss}

    def cont_loss_func(self, recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> dict:
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        total_loss = recon_loss + kld_loss
        
        return {'total_loss': total_loss, 'reconstruction_loss': recon_loss, 'kld_loss': kld_loss}
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x, mask)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return recon_x, mu, log_var


if __name__ == "__main__":
    e_cfg = EncConfig()
    d_cfg = DecConfig()
    attn = AutoEncoder(e_cfg, d_cfg)
    # latents = torch.randn((3, 16, 1024)).cuda()
    x = torch.randn((3, 300, 512)).cuda()

    print(attn(x).shape) 
