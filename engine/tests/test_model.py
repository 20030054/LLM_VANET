"""The layer-wise forward pass must reproduce Hugging Face exactly."""
import torch
from kvrsu.model import load_model


def test_forward_matches_huggingface():
    model, _, _ = load_model("tiny-random", "cpu")
    torch.manual_seed(1)
    ids = torch.randint(0, 256, (3, 40))
    last = torch.tensor([39, 20, 5])
    ours = model.forward_full(ids, last)
    with torch.no_grad():
        ref = model.hf(ids).logits[torch.arange(3), last]
    assert torch.allclose(ours, ref, atol=1e-5), (ours - ref).abs().max()


def test_incremental_equals_full():
    """Processing a sequence in two pieces with cached KV gives the same result."""
    model, _, _ = load_model("tiny-random", "cpu")
    torch.manual_seed(2)
    ids = torch.randint(0, 256, (1, 30))
    full = model.forward_full(ids, torch.tensor([29]))
    with torch.no_grad():
        h1, h2 = model.embed_tokens(ids[:, :18]), model.embed_tokens(ids[:, 18:])
        for l in range(model.n_layers):
            h1, k, v = model.layer_step(l, h1, torch.arange(18))
            h2, _, _ = model.layer_step(l, h2, torch.arange(18, 30), k, v)
        part = model.head(h2[:, -1])
    assert torch.allclose(full, part, atol=1e-5)
