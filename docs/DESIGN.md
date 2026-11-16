# Design notes

## Problem (per decision window w)

Shared table of `p` tokens, of which the first `u` are unchanged since the last served window
(`u` = longest common token prefix), `ghat` of those resident on the GPU. Serve `n` requests with
`s` suffix tokens each. Choose `m` (`ghat ≤ m ≤ u`) cached tokens to use; recompute `p − m`.

    T_cmp  = t(p − m, p) + t(n·s, p + s)        t(N, c) = a1·N + a2·N·c
    T_load = κ·(m − ghat)/B_cg                  κ = KV bytes per token per layer
    T_off  = κ·(p − m)/B_gc
    T_srv  = L·max(T_cmp, T_load, T_off) + t0

Maximise the number of requests whose decision arrives within their slack, subject to the GPU KV
budget. Solved exactly by a scan (`costmodel.py`): for each batch size `n` (largest slack first) pick
the best `m`; admit the largest `n` whose predicted `T_srv` fits the `n`-th slack. Non-admitted
requests get the weighted heuristic at once.

**Ordering (Proposition 1).** If slot `i` changes with probability `q_i`, independently, the expected
reusable prefix is `Σ_k Π_{i≤k}(1 − q_i)`, maximal for non-decreasing `q`. `TableManager` estimates
`q` by an exponential average of observed changes and re-sorts every 20 windows only if the expected
gain over that period exceeds the one-off loss from moving slots.

## Where the code differs from the written formulation

| Formulation | Code | Why |
|---|---|---|
| change probability from the TinyLSTM link predictor | EWMA of observed slot changes | no trained predictor needed; a predictor can replace `TableManager.q` |
| fixed-width slots of ℓ tokens | fixed text format, tokenised per line; reuse measured in tokens | token counts per line may differ slightly; the LCP is exact anyway |
| memory term `κ·n·s` for the batch | transient `κ·c·(p+s)` per layer for a chunk of `c ≤ 32` requests | the shared prefix is expanded per chunk for the attention kernel |
| stability constraint `T_srv ≤ Δ` | optional (`--max-service`); a busy engine delays the next window | lets the baselines show their backlog |
| slack from deadline and link lifetime | deadline minus age minus best estimated completion time | link lifetime enters through option feasibility |

## Engine

`model.py` runs one decoder layer at a time on Hugging Face's own weight modules. Grouped-query
attention folds the query heads of a group into the query-length axis, so keys/values are never
repeated. `serving.py` interleaves, per layer: wait for the prefetched cache block of this layer,
start prefetching the next layer, recompute the table tail, offload its new KV, serve the request
batch, keep the first `g` tokens resident. Copies use separate CUDA streams and pinned host memory.

## Simulation

Vehicles report state every 0.5 s (broadcast) and send one unicast request per task. The RSU batches
requests per 100 ms window, calls the server, and sends each decision after the measured inference
time (`computeScale` multiplies it). Task input travels as real 802.11p frames (1300-byte fragments,
MAC ACK/retry); executors have a FIFO CPU queue. A task succeeds if its result is back before the
deadline. No decision within half the deadline ⇒ the vehicle runs the task locally (`timeout`).

## Known limitations (state them in the paper)

- One RSU, one collision domain, synthetic grid; no multi-RSU handover.
- Hindsight labels use future positions and the queue length at decision time, not future queues.
- The KV budget and slower hardware are emulated on an HPC GPU.
- The model is 1B parameters; its weights and, at these table sizes, often its cache fit a modern
  GPU. E5 shows from which budget on spilling matters. If it never matters, say so and present
  spilling as relevant to smaller devices or larger models only.
- ~10 % of tasks arise outside reliable RSU coverage and run locally (`nocoverage`), for all policies.
