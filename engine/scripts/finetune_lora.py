#!/usr/bin/env python3
"""Fine-tune the decision LLM with LoRA and save a merged model.

    python engine/scripts/finetune_lora.py --data data/train.jsonl \
        --model meta-llama/Llama-3.2-1B --out models/llama32-1b-kvrsu

The model is trained to output the answer letter as the next token after the
prompt (cross-entropy on that single position, as in the Sci. Reports 2026
paper).  Prompts are tokenised exactly as the serving engine does it: header,
each table line and the suffix are tokenised separately and concatenated.
The LoRA weights are merged before saving, so the result loads as an
ordinary model in the decision server (``--model models/llama32-1b-kvrsu``).
"""
import argparse
import json
import random

import _common  # noqa: F401
import torch
import torch.nn.functional as F

from kvrsu.prompt import HEADER, ALL_LABELS, LabelSet


def load(name, device, dtype):
    if name == "tiny-random":                      # smoke test without download
        from kvrsu.model import load_model
        m, encode, bos = load_model(name, device)
        return m.hf, None, encode, bos
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    hf = AutoModelForCausalLM.from_pretrained(name, torch_dtype=getattr(torch, dtype)).to(device)
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    return hf, tok, (lambda s: tok(s, add_special_tokens=False).input_ids), bos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--holdout", type=float, default=0.05)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)

    from peft import LoraConfig, get_peft_model
    hf, tok, encode, bos = load(a.model, a.device, a.dtype)
    labels = LabelSet(encode)
    label_ids = torch.tensor([labels.ids[l] for l in ALL_LABELS], device=a.device)
    model = get_peft_model(hf, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()

    with open(a.data) as f:
        raw = [json.loads(line) for line in f]
    if a.max_samples:
        raw = raw[: a.max_samples]
    head = [bos] + encode(HEADER)
    cache = {}
    def tokens(s):
        ids = list(head)
        for line in s["entries"]:
            if line not in cache:
                cache[line] = encode(line)
            ids += cache[line]
        return ids + encode(s["suffix"]), ALL_LABELS.index(s["label"])
    data = [tokens(s) for s in raw]
    cut = max(1, int(len(data) * (1 - a.holdout)))
    train, held = data[:cut], data[cut:]

    def batch_logits(batch):
        n = max(len(x) for x, _ in batch)
        ids = torch.zeros(len(batch), n, dtype=torch.long)
        mask = torch.zeros(len(batch), n, dtype=torch.long)
        for i, (x, _) in enumerate(batch):
            ids[i, : len(x)] = torch.tensor(x); mask[i, : len(x)] = 1
        out = model(input_ids=ids.to(a.device), attention_mask=mask.to(a.device)).logits
        last = torch.tensor([len(x) - 1 for x, _ in batch], device=a.device)
        logits = out[torch.arange(len(batch), device=a.device), last].float()[:, label_ids]   # only the answer letters
        return logits, torch.tensor([y for _, y in batch], device=a.device)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    steps = int(a.epochs * len(train) / a.batch)
    model.train()
    for step in range(steps):
        logits, y = batch_logits(random.sample(train, min(a.batch, len(train))))
        loss = F.cross_entropy(logits, y) / a.accum
        loss.backward()
        if (step + 1) % a.accum == 0 or step == steps - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()
        if step % 50 == 0:
            print(f"step {step}/{steps} loss {loss.item() * a.accum:.4f}", flush=True)

    model.eval()
    hits = 0
    with torch.no_grad():
        for i in range(0, len(held), a.batch):
            logits, y = batch_logits(held[i:i + a.batch])
            hits += (logits.argmax(-1) == y).sum().item()
    print(f"held-out accuracy {hits / max(1, len(held)):.3f} on {len(held)} requests")

    merged = model.merge_and_unload()
    merged.save_pretrained(a.out)
    if tok is not None:
        tok.save_pretrained(a.out)
    print("merged model saved to", a.out)


if __name__ == "__main__":
    main()
