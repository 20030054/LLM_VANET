"""Layer-wise Llama forward pass with explicit control of the key/value cache.

Hugging Face's ``model(...)`` call runs all layers at once and hides the KV
cache behind a ``Cache`` object.  The serving engine needs the opposite: run
ONE layer at a time, decide per layer which cached keys/values come from GPU
memory, which are loaded from host memory, and which are recomputed.

``LayerwiseLlama`` therefore re-implements only the *control flow* of a
Llama-family decoder.  All weights stay inside the original Hugging Face
modules (``q_proj``, ``mlp`` ...), so any checkpoint that loads as
``LlamaForCausalLM`` works, including LoRA-merged and 4-bit models.

Numerical equivalence with the Hugging Face forward pass is checked by
``engine/tests/test_model.py`` and ``engine/scripts/check_equivalence.py``.
Run the latter once on every new model / library version.

Tensor layout
-------------
Keys and values of one layer are stored as ``(batch, n_kv_heads, tokens,
head_dim)``.  Keys are stored *after* the rotary position embedding, exactly
like Hugging Face does, so a cached key is only valid at the position it was
computed for.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class LayerwiseLlama:
    """Runs a Llama-family causal LM one layer at a time."""

    def __init__(self, hf_model):
        self.hf = hf_model.eval()
        cfg = hf_model.config
        base = hf_model.model
        self.layers = base.layers
        self.embed = base.embed_tokens
        self.norm = base.norm
        self.lm_head = hf_model.lm_head

        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.n_kv = getattr(cfg, "num_key_value_heads", None) or self.n_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.n_heads
        self.groups = self.n_heads // self.n_kv
        self.hidden = cfg.hidden_size
        self.vocab = cfg.vocab_size

        self.device = self.embed.weight.device
        self.dtype = self.embed.weight.dtype
        self.inv_freq = self._find_inv_freq(cfg, base).to(self.device, torch.float32)

    # ------------------------------------------------------------------ sizes
    @property
    def kv_bytes_per_token_layer(self) -> int:
        """kappa in the paper: bytes of K plus V for one token in one layer."""
        return 2 * self.n_kv * self.head_dim * torch.finfo(self.dtype).bits // 8

    # ------------------------------------------------------------------- RoPE
    @staticmethod
    def _find_inv_freq(cfg, base) -> torch.Tensor:
        """Take the rotary frequencies from Hugging Face so that scaled
        variants (e.g. ``rope_type="llama3"``) are handled identically."""
        rot = getattr(base, "rotary_emb", None)
        if rot is None:  # older transformers: one rotary module per layer
            rot = getattr(base.layers[0].self_attn, "rotary_emb", None)
        if rot is not None and hasattr(rot, "inv_freq"):
            return rot.inv_freq.detach().clone()
        dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        theta = getattr(cfg, "rope_theta", 10000.0)
        return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    def _rope(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding. x: (B, H, T, head_dim), pos: (T,)."""
        freqs = pos.to(torch.float32)[:, None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(x.dtype)[None, None]
        sin = emb.sin().to(x.dtype)[None, None]
        half = x.shape[-1] // 2
        rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        return x * cos + rotated * sin

    # -------------------------------------------------------------- one layer
    def layer_step(
        self,
        layer_idx: int,
        h: torch.Tensor,
        pos: torch.Tensor,
        past_k: Optional[torch.Tensor] = None,
        past_v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one decoder layer on ``T`` new tokens.

        Args:
            h:      (B, T, hidden) hidden states of the new tokens.
            pos:    (T,) absolute positions of the new tokens.
            past_k: (1 or B, n_kv, P, head_dim) keys of earlier tokens, or None.
            past_v: same shape, values.

        Every new token attends to all ``P`` past tokens and causally to the
        new tokens.  A past with batch size 1 is shared by the whole batch.

        Returns ``(h_out, k_new, v_new)`` where ``k_new``/``v_new`` are the
        keys/values of the new tokens only, shape (B, n_kv, T, head_dim).
        """
        layer = self.layers[layer_idx]
        attn = layer.self_attn
        B, T, _ = h.shape

        x = layer.input_layernorm(h)
        q = attn.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = attn.k_proj(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        v = attn.v_proj(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        q = self._rope(q, pos)
        k = self._rope(k, pos)

        if past_k is not None and past_k.shape[2] > 0:
            P = past_k.shape[2]
            if past_k.shape[0] != B:
                past_k = past_k.expand(B, -1, -1, -1)
                past_v = past_v.expand(B, -1, -1, -1)
            keys = torch.cat((past_k, k), dim=2)
            vals = torch.cat((past_v, v), dim=2)
        else:
            P = 0
            keys, vals = k, v

        # Grouped-query attention without copying K/V: the `groups` query
        # heads that share one KV head are folded into the query-length axis.
        g = self.groups
        q = q.reshape(B, self.n_kv, g * T, self.head_dim)
        mask = torch.ones(T, P + T, dtype=torch.bool, device=h.device)
        mask[:, P:] = torch.tril(mask[:, P:])
        out = F.scaled_dot_product_attention(q, keys, vals, attn_mask=mask.repeat(g, 1))
        out = out.reshape(B, self.n_heads, T, self.head_dim).transpose(1, 2).reshape(B, T, -1)

        h = h + attn.o_proj(out)
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        return h, k, v

    # ------------------------------------------------------------ whole model
    def embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embed(ids)

    def head(self, h_last: torch.Tensor) -> torch.Tensor:
        """Final norm + LM head on (B, hidden) hidden states -> (B, vocab)."""
        return self.lm_head(self.norm(h_last))

    @torch.no_grad()
    def forward_full(self, ids: torch.Tensor, last_idx: torch.Tensor) -> torch.Tensor:
        """Plain forward pass without any cache.

        Args:
            ids:      (B, T) token ids, right-padded.
            last_idx: (B,) index of the last real token of each row.

        Returns next-token logits (B, vocab) at ``last_idx``.  Right padding
        needs no mask: with causal attention a token never sees later ones.
        """
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.embed(ids)
        for i in range(self.n_layers):
            h, _, _ = self.layer_step(i, h, pos)
        h_last = h[torch.arange(B, device=ids.device), last_idx]
        return self.head(h_last)


def load_model(name_or_path: str, device: str = "cuda", dtype: str = "float16", load_4bit: bool = False):
    """Load a model and tokenizer for the engine.

    ``name_or_path = "tiny-random"`` builds a small random Llama with a byte
    tokenizer.  It needs no download and no GPU and is used by the tests and
    by the local smoke test of the simulation.

    Returns ``(LayerwiseLlama, encode, bos_id)`` where ``encode(text)`` gives
    a list of token ids without special tokens.
    """
    if name_or_path == "tiny-random":
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(0)
        cfg = LlamaConfig(
            vocab_size=260, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
            num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=65536,
        )
        hf = LlamaForCausalLM(cfg).to(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        return LayerwiseLlama(hf), (lambda text: list(text.encode("ascii", "replace"))), 257

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name_or_path)
    kwargs = {"torch_dtype": getattr(torch, dtype)}
    if load_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=getattr(torch, dtype), bnb_4bit_quant_type="nf4"
        )
        kwargs["device_map"] = {"": device}
    hf = AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs)
    if not load_4bit:
        hf = hf.to(device)
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    return LayerwiseLlama(hf), (lambda text: tok(text, add_special_tokens=False).input_ids), bos
