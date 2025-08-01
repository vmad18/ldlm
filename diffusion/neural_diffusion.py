import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from einops import rearrange, repeat
from math import sqrt, log

from typing import Tuple, Optional


class DiTConfig:
    num_latents = 32 
    latent_dim = 768         
    
    dim = 512              
    num_layers = 12
    num_heads = 8 
    expansion_factor = 4
    
    rope_base = int(1e5)

    # max_seq_len = 64

    dev = "cuda" if torch.cuda.is_available() else "cpu"


class RoPE(nn.Module):

    def __init__(self, dim: int, cfg: DiTConfig):
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even for RoPE."
        
        theta = 1.0 / (cfg.rope_base ** (torch.arange(0, dim, 2).float() / dim))
        seq_idx = torch.arange(cfg.num_latents).float()
        
        idx_theta = torch.einsum("n,d->nd", seq_idx, theta)
        freqs_cis = torch.polar(torch.ones_like(idx_theta), idx_theta)
        
        self.register_buffer("freqs_cis", freqs_cis)

    def _to_complex(self, x: torch.Tensor) -> torch.Tensor:
        return torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    def _to_real(self, x: torch.Tensor) -> torch.Tensor:
        return torch.view_as_real(x).flatten(start_dim=-2)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        seq_len_q = q.shape[-2]
        seq_len_k = k.shape[-2]
        
        assert seq_len_q <= self.freqs_cis.shape[0], f"Query sequence length {seq_len_q} exceeds RoPE's max length {self.freqs_cis.shape[0]}"
        assert seq_len_k <= self.freqs_cis.shape[0], f"Key sequence length {seq_len_k} exceeds RoPE's max length {self.freqs_cis.shape[0]}"

        rope_q = self.freqs_cis[:seq_len_q].view(1, 1, seq_len_q, -1)
        rope_k = self.freqs_cis[:seq_len_k].view(1, 1, seq_len_k, -1)
        
        q_rotated = self._to_complex(q) * rope_q
        k_rotated = self._to_complex(k) * rope_k
        
        q_out = self._to_real(q_rotated).to(q.dtype)
        k_out = self._to_real(k_rotated).to(k.dtype)
        return q_out, k_out


class TimestepEmbedder(nn.Module):

    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cfg.dim, int(cfg.dim * cfg.expansion_factor)),
            nn.SiLU(),
            nn.Linear(int(cfg.dim * cfg.expansion_factor), cfg.dim),
        )
        self.frequency_embedding_size = cfg.dim

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period=10000) -> torch.Tensor:

        half = dim // 2
        freqs = torch.exp(
            -log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


# TODO fuse the two Attns to be Perceiver-Styled
class SelfAttention(nn.Module):
    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.num_heads, self.head_dim = cfg.num_heads, cfg.dim // cfg.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.proj_o = nn.Linear(cfg.dim, cfg.dim)

        self.rope = RoPE(dim=self.head_dim, cfg=cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, *_ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        q, k = self.rope(q, k)
        
        sim = (q @ k.transpose(-2, -1)) * self.scale
        attn = sim.softmax(dim=-1) @ v
        x = rearrange(attn, 'b h n d -> b n (h d)')
        return self.proj_o(x)


class CrossAttention(nn.Module):
    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.num_heads, self.head_dim = cfg.num_heads, cfg.dim // cfg.num_heads
        self.scale = self.head_dim ** -0.5
        
        self.proj_q = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.proj_kv = nn.Linear(cfg.dim, 2 * cfg.dim, bias=False)
        
        self.proj_o = nn.Linear(cfg.dim, cfg.dim)

        self.rope = RoPE(dim=self.head_dim, cfg=cfg)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        # Project and reshape q, k, v
        q = self.proj_q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.proj_kv(ctx).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        q, k = self.rope(q, k)
        
        sim = (q @ k.transpose(-2, -1)) * self.scale
        attn = sim.softmax(dim = -1) @ v
        x = rearrange(attn, 'b h n d -> b n (h d)')
        return self.proj_o(x)


class FeedForward(nn.Module):

    def __init__(self, cfg: DiTConfig) -> None:
        super().__init__()
        hidden_dim = int(cfg.dim * cfg.expansion_factor)
        self.proj_up = nn.Linear(cfg.dim, hidden_dim) 
        self.proj_down = nn.Linear(hidden_dim, cfg.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.proj_down(torch.nn.functional.gelu(self.proj_up(x))) # seems that gelu is >> better than relu^2


class FinalLayer(nn.Module):
    """ The final layer of DiT. """
    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.norm_final = nn.LayerNorm(cfg.dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(cfg.dim, cfg.latent_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cfg.dim, 2 * cfg.dim, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.proj_out(x)
        return x


def modulate(x, shift, scale):
    """ modulates the input tensor `x` using learned shift and scale. """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """ DiT block with self-attention, cross-attention, and adaLN modulation """
    def __init__(self, cfg: DiTConfig):
        super().__init__()


        # time embedding modulations 
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cfg.dim, 4 * cfg.dim, bias=True) # 6 for shift/scale for 3 LayerNorms
        )

        # norms
        self.attn_norm = nn.LayerNorm(cfg.dim, elementwise_affine=False) 
        # self.cattn_norm = nn.LayerNorm(cfg.dim, elementwise_affine=False)
        self.ff_norm = nn.LayerNorm(cfg.dim, elementwise_affine=False) 

        self.attn = SelfAttention(cfg)
        # self.cross_attn = CrossAttention(cfg)
        self.ff = FeedForward(cfg)

    def forward(self, 
                x: torch.Tensor, 
                t: torch.Tensor,) -> torch.Tensor:
        shift_msa, scale_msa, shift_ffn, scale_ffn = self.adaLN_modulation(t).chunk(4, dim=1)
        x = x + self.attn(modulate(self.attn_norm(x), shift_msa, scale_msa))
        # x = x + self.cross_attn(modulate(self.cattn_norm(x), shift_ca, scale_ca), ctx) # Pass context here
        x = x + self.ff(modulate(self.ff_norm(x), shift_ffn, scale_ffn))
        return x
        

class DiTModel(nn.Module):

    def __init__(self, 
                 cfg: DiTConfig,
                 class_conditional = False,
                 self_condition = False,
                 class_unconditional_prob = 0, 
                 num_classes = 0,  
                 seq2seq = False, 
                 ):
        super().__init__()
        self.cfg = cfg
        
        self.class_conditional = class_conditional
        self.self_condition = self_condition
        self.class_unconditional_prob = class_unconditional_prob
        self.num_classes = num_classes
        self.seq2seq = seq2seq

        self.proj_in = nn.Linear(cfg.latent_dim, cfg.dim)
        # self.ctx_proj = nn.Linear(cfg.latent_dim, cfg.dim)

        self.t_embed = TimestepEmbedder(cfg)
        
        # self.null_context = nn.Parameter(torch.randn(1, cfg.num_latents, cfg.latent_dim))
        
        self.blocks = nn.ModuleList([DiTBlock(cfg) for _ in range(cfg.num_layers)])
        self.proj_out = FinalLayer(cfg)

    # TODO: change signature to handle ctx tensor: ctx: torch.Tensor
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1) 
        # ctx = repeat(self.null_context, '1 n d -> b n d', b=x.shape[0])
        
        x = self.proj_in(x)
        # ctx = self.ctx_proj(ctx)
        
        t = self.t_embed(t)
        for block in self.blocks:
            x = block(x, t)

        x = self.proj_out(x, t)        
        return x



if __name__ == '__main__':
    config = DiTConfig()
    model = DiTModel(config).to(config.dev)
    
    # Create a dummy input batch
    x_in = torch.randn(16, config.num_latents, config.latent_dim, device=config.dev)
    
    ctx = torch.randn_like(x_in)

    # Create a dummy timestep batch
    t_in = torch.randint(0, 1000, (16,), device=config.dev)
    
    # Forward pass
    predicted_noise = model(x_in, t_in, ctx)
    
    print(f"Model Input Shape:  {x_in.shape}")
    print(f"Model Output Shape: {predicted_noise.shape}")
    
    assert x_in.shape == predicted_noise.shape
    print("\nModel instantiated and forward pass successful!")
