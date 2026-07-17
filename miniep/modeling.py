"""From-scratch GraniteMoE implementation (inference only, fp32-friendly).

Mirrors the HF `granitemoe` architecture exactly:
  - RMSNorm (eps from config), no biases anywhere, tied embeddings
  - embeddings scaled by `embedding_multiplier`; final logits divided by `logits_scaling`
  - residual updates scaled by `residual_multiplier`
  - attention scores scaled by `attention_multiplier` (not 1/sqrt(d_head)); GQA; llama RoPE
  - router: linear -> top-k logits -> softmax over the top-k only (fp32)
  - experts: fused gate+up `input_linear` (E, 2I, H), silu(gate)*up, `output_linear` (E, H, I)

Expert execution is delegated to a pluggable MoE backend so the same transformer
runs single-process (LocalMoEBackend) or expert-parallel (ep.EPMoEBackend).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    num_local_experts: int
    num_experts_per_tok: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    embedding_multiplier: float
    attention_multiplier: float
    residual_multiplier: float
    logits_scaling: float
    max_position_embeddings: int
    # granite-3.0: eos == pad == 0; padded positions are masked out of both
    # attention and MoE dispatch, so pad-as-eos never leaks into real outputs
    eos_token_id: int = 0
    pad_token_id: int = 0

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    @staticmethod
    def from_json(path: str | Path) -> "ModelConfig":
        raw = json.loads(Path(path).read_text())
        return ModelConfig(**{k: raw[k] for k in ModelConfig.__dataclass_fields__})


def rms_norm(x, weight, eps):
    return weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Rope:
    def __init__(self, cfg: ModelConfig):
        d = cfg.head_dim
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
        t = torch.arange(cfg.max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos = emb.cos()  # (max_pos, head_dim)
        self.sin = emb.sin()

    def apply(self, q, k, positions):
        # q: (B, Hq, T, D), k: (B, Hk, T, D), positions: (B, T) absolute positions
        cos = self.cos[positions].unsqueeze(1)  # (B, 1, T, D)
        sin = self.sin[positions].unsqueeze(1)
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def run_experts(x, expert_ids, w_in, w_out, timings=None):
    """Run each token-entry through its assigned expert.

    x: (M, H) hidden states, one row per (token, expert-slot) entry
    expert_ids: (M,) which expert each row goes to (indices into w_in/w_out dim 0)
    w_in: (E, 2I, H) fused gate+up; w_out: (E, H, I)
    Returns (M, H). Rows are processed grouped by expert; output keeps input order.
    """
    out = torch.zeros(x.shape[0], w_out.shape[1], dtype=x.dtype, device=x.device)
    if x.shape[0] == 0:
        return out
    order = torch.argsort(expert_ids, stable=True)
    sorted_ids = expert_ids[order]
    uniq, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
    start = 0
    for e, n in zip(uniq.tolist(), counts.tolist()):
        idx = order[start:start + n]
        start += n
        h = x[idx] @ w_in[e].T
        gate, up = h.chunk(2, dim=-1)
        out[idx] = (F.silu(gate) * up) @ w_out[e].T
    return out


def flatten_topk(n, k, valid_mask=None):
    """Expand n tokens x k routing slots into flat per-entry indices.

    Returns (token_ids, slot_ids, keep_e): keep_e selects valid entries from any
    reshaped (n*k) tensor; token_ids/slot_ids give each kept entry's (token, slot).
    """
    keep = torch.ones(n, dtype=torch.bool) if valid_mask is None else valid_mask
    keep_e = keep.repeat_interleave(k)
    token_ids = torch.arange(n).repeat_interleave(k)[keep_e]
    slot_ids = torch.arange(k).repeat(n)[keep_e]
    return token_ids, slot_ids, keep_e


def scatter_combine(gated, token_ids, slot_ids, n, k, hidden):
    """Sum each token's k gated expert outputs. gated: (M, H), one row per entry.

    Each (token, slot) cell is unique, so this scatter needs no atomic adds
    (unlike index_add_) and stays deterministic on CUDA regardless of placement.
    """
    buf = torch.zeros(n, k, hidden, dtype=gated.dtype, device=gated.device)
    buf[token_ids, slot_ids] = gated
    return buf.sum(dim=1)


class LocalMoEBackend:
    """All experts resident in-process; reference execution path."""

    def __init__(self, w_in, w_out):
        self.w_in = w_in    # (E, 2I, H)
        self.w_out = w_out  # (E, H, I)

    def __call__(self, x_flat, topk_idx, topk_gate, valid_mask=None):
        # x_flat: (N, H); topk_idx/topk_gate: (N, k); valid_mask: (N,) bool or None
        n, k = topk_idx.shape
        token_ids, slot_ids, keep_e = flatten_topk(n, k, valid_mask)
        expert_ids = topk_idx.reshape(-1)[keep_e]
        outs = run_experts(x_flat[token_ids], expert_ids, self.w_in, self.w_out)
        gated = outs * topk_gate.reshape(-1)[keep_e].unsqueeze(1)
        return scatter_combine(gated, token_ids, slot_ids, n, k, x_flat.shape[1])


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, weights: dict, layer: int, moe_backend):
        super().__init__()
        self.cfg = cfg
        p = f"model.layers.{layer}."
        self.input_ln = weights[p + "input_layernorm.weight"]
        self.post_ln = weights[p + "post_attention_layernorm.weight"]
        self.wq = weights[p + "self_attn.q_proj.weight"]
        self.wk = weights[p + "self_attn.k_proj.weight"]
        self.wv = weights[p + "self_attn.v_proj.weight"]
        self.wo = weights[p + "self_attn.o_proj.weight"]
        self.router = weights[p + "block_sparse_moe.router.layer.weight"]
        self.moe = moe_backend
        self.layer = layer

    def attention(self, x, rope, positions, kv_cache, attn_mask):
        cfg = self.cfg
        B, T, H = x.shape
        d = cfg.head_dim
        q = (x @ self.wq.T).view(B, T, cfg.num_attention_heads, d).transpose(1, 2)
        k = (x @ self.wk.T).view(B, T, cfg.num_key_value_heads, d).transpose(1, 2)
        v = (x @ self.wv.T).view(B, T, cfg.num_key_value_heads, d).transpose(1, 2)
        q, k = rope.apply(q, k, positions)
        if kv_cache is not None:
            k, v = kv_cache.update(self.layer, k, v)
        rep = cfg.num_attention_heads // cfg.num_key_value_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        # with no explicit mask: causal for multi-token prefill; a single decode
        # query attends the whole cache (sdpa's is_causal would top-left-align
        # the mask and wrongly restrict a 1-token query to the first key)
        o = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, scale=cfg.attention_multiplier,
            is_causal=attn_mask is None and T > 1,
        )
        o = o.transpose(1, 2).reshape(B, T, H)
        return o @ self.wo.T

    def forward(self, h, rope, positions, kv_cache, attn_mask, moe_valid=None):
        cfg = self.cfg
        x = rms_norm(h, self.input_ln, cfg.rms_norm_eps)
        h = h + self.attention(x, rope, positions, kv_cache, attn_mask) * cfg.residual_multiplier
        x = rms_norm(h, self.post_ln, cfg.rms_norm_eps)
        B, T, H = x.shape
        x_flat = x.reshape(-1, H)
        router_logits = x_flat @ self.router.T
        topk_logits, topk_idx = router_logits.topk(cfg.num_experts_per_tok, dim=-1)
        topk_gate = torch.softmax(topk_logits.float(), dim=-1).to(x.dtype)
        moe_out = self.moe(x_flat, topk_idx, topk_gate, valid_mask=moe_valid).reshape(B, T, H)
        return h + moe_out * cfg.residual_multiplier


class KVCache:
    def __init__(self, num_layers):
        self.k = [None] * num_layers
        self.v = [None] * num_layers

    def update(self, layer, k, v):
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k, v
        else:
            self.k[layer] = torch.cat([self.k[layer], k], dim=2)
            self.v[layer] = torch.cat([self.v[layer], v], dim=2)
        return self.k[layer], self.v[layer]

    @property
    def seq_len(self):
        return 0 if self.k[0] is None else self.k[0].shape[2]


class GraniteMoeModel(nn.Module):
    """moe_backends: one callable per layer (see LocalMoEBackend for the contract)."""

    def __init__(self, cfg: ModelConfig, weights: dict, moe_backends: list):
        super().__init__()
        self.cfg = cfg
        self.embed = weights["model.embed_tokens.weight"]
        self.final_norm = weights["model.norm.weight"]
        self.rope = Rope(cfg)
        self.blocks = [Block(cfg, weights, i, moe_backends[i]) for i in range(cfg.num_hidden_layers)]

    def forward(self, input_ids, positions=None, kv_cache=None, attn_mask=None,
                moe_valid=None, logits_at=None):
        """logits_at: optional (B,) per-row position; compute logits only for
        that position of each row (a large saving on long prefills; the full
        (B,T,V) lm_head GEMM is otherwise the biggest serial cost)."""
        B, T = input_ids.shape
        if positions is None:
            start = kv_cache.seq_len if kv_cache is not None else 0
            positions = torch.arange(start, start + T).unsqueeze(0).expand(B, T)
        if moe_valid is not None:
            moe_valid = moe_valid.reshape(-1)
        h = self.embed[input_ids] * self.cfg.embedding_multiplier
        for blk in self.blocks:
            h = blk(h, self.rope, positions, kv_cache, attn_mask, moe_valid=moe_valid)
        if logits_at is not None:
            h = h[torch.arange(B), logits_at].unsqueeze(1)  # (B, 1, H)
        h = rms_norm(h, self.final_norm, self.cfg.rms_norm_eps)
        return (h @ self.embed.T) / self.cfg.logits_scaling


def greedy_generate(model, input_ids, max_new_tokens, eos_id=None):
    """input_ids: (1, T). Returns generated ids (list). Single-sequence utility."""
    kv = KVCache(model.cfg.num_hidden_layers)
    logits = model(input_ids, kv_cache=kv)
    out = []
    tok = logits[0, -1].argmax().item()
    for _ in range(max_new_tokens):
        out.append(tok)
        if eos_id is not None and tok == eos_id:
            break
        step = torch.tensor([[tok]])
        logits = model(step, kv_cache=kv)
        tok = logits[0, -1].argmax().item()
    return out
