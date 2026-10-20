"""AgmGPT adapter over the pinned, unmodified Karpathy nanochat model."""
import os
import sys
from dataclasses import dataclass
from pathlib import Path
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')
os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')  # Windows: no Triton required
sys.path.insert(0, str(Path(__file__).resolve().parent / 'vendor' / 'nanochat'))
import torch
from torch.utils.checkpoint import checkpoint
from nanochat.gpt import GPT as NanoGPT, GPTConfig as NanoConfig
from nanochat.common import COMPUTE_DTYPE


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    block_size: int = 1024
    n_layer: int = 8
    n_head: int = 8
    n_kv_head: int = 4
    n_embd: int = 512
    window_pattern: str = 'SSSL'
    gradient_checkpointing: bool = True

    def __post_init__(self):
        if min(self.vocab_size, self.block_size, self.n_layer, self.n_head, self.n_kv_head, self.n_embd) <= 0:
            raise ValueError('Model dimensions must be positive')
        if self.n_embd < 24 or self.n_embd % self.n_head or (self.n_embd // self.n_head) % 2:
            raise ValueError('Width must be >=24 with an even head dimension')
        if self.n_head % self.n_kv_head or not self.window_pattern or set(self.window_pattern) - set('SL'):
            raise ValueError('Invalid GQA heads or attention window pattern')

    def upstream(self):
        return NanoConfig(sequence_len=self.block_size, vocab_size=self.vocab_size,
                          n_layer=self.n_layer, n_head=self.n_head, n_kv_head=self.n_kv_head,
                          n_embd=self.n_embd, window_pattern=self.window_pattern)


class CheckpointBlock(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, *args):
        if self.training and torch.is_grad_enabled():
            return checkpoint(self.block, *args, use_reentrant=False)
        return self.block(*args)


class KVCache:
    """Upstream cache protocol without importing the upstream tokenizer stack."""
    def __init__(self, c, batch, length, device):
        self.n_layers = c.n_layer
        shape = (c.n_layer, batch, length, c.n_kv_head, c.n_embd // c.n_head)
        self.k_cache = torch.zeros(shape, device=device, dtype=COMPUTE_DTYPE)
        self.v_cache = torch.zeros_like(self.k_cache)
        self.cache_seqlens = torch.zeros(batch, device=device, dtype=torch.int32)
        self.prev_embedding = None

    def get_pos(self):
        return int(self.cache_seqlens[0])

    def get_layer_cache(self, i):
        return self.k_cache[i], self.v_cache[i]

    def advance(self, count):
        self.cache_seqlens += count


class GPT(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.core = NanoGPT(config.upstream())
        self.core.init_weights()
        if config.gradient_checkpointing:
            self.core.transformer.h = torch.nn.ModuleList(CheckpointBlock(b) for b in self.core.transformer.h)

    def forward(self, idx, targets=None, kv_cache=None):
        if idx.ndim != 2 or not 0 < idx.size(1) <= self.config.block_size:
            raise ValueError('Input exceeds context window')
        if targets is not None:
            if targets.shape != idx.shape or not bool(targets.ne(-1).any()):
                raise ValueError('Targets must contain supervised tokens')
            return None, self.core(idx, targets, kv_cache=kv_cache)
        return self.core(idx, kv_cache=kv_cache), None

    def setup_optimizer(self, kind='muon', lr=3e-4, weight_decay=.1):
        if kind == 'muon':
            return self.core.setup_optimizer(weight_decay=weight_decay)
        decay, no_decay = [], []
        for p in self.parameters():
            (decay if p.ndim >= 2 else no_decay).append(p)
        return torch.optim.AdamW([{'params': decay, 'weight_decay': weight_decay},
                                  {'params': no_decay, 'weight_decay': 0.0}], lr=lr, betas=(.9, .95))

    @torch.inference_mode()
    def generate_stream(self, idx, count=160, temperature=.8, top_k=40, stop_ids=()):
        if idx.size(0) != 1 or count < 0 or temperature < 0 or top_k < 0:
            raise ValueError('Invalid generation arguments')
        self.eval()
        idx = idx[:, -self.config.block_size:]
        cache = KVCache(self.config, 1, self.config.block_size, idx.device)
        current = idx
        for _ in range(count):
            if cache.get_pos() + current.size(1) > self.config.block_size:
                cache = KVCache(self.config, 1, self.config.block_size, idx.device)
                current = idx[:, -self.config.block_size:]
            logits, _ = self(current, kv_cache=cache)
            scores = logits[:, -1].float()
            if temperature == 0:
                token = scores.argmax(-1, keepdim=True)
            else:
                scores = scores / temperature
                if top_k:
                    cut = torch.topk(scores, min(top_k, scores.size(-1))).values[:, -1:]
                    scores = scores.masked_fill(scores < cut, -float('inf'))
                token = torch.multinomial(scores.softmax(-1), 1)
            value = int(token.item())
            if value in stop_ids:
                break
            yield value
            idx = torch.cat((idx, token), dim=1)
            current = token


def size_report(config):
    with torch.device('meta'):
        n = sum(p.numel() for p in NanoGPT(config.upstream()).parameters())
    return dict(parameters=n, float_parameters=n, binary_parameters=0,
                fp32_weight_bytes=n * 4, estimated_adam_training_bytes=n * 16)
