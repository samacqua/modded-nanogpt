# TTT LoRA Progress

Checkpoint: `logs/ddf165fd-4ed9-43d8-aa3c-23332ae04564/state_pre_ttt.pt`
Hardware: 2×H100, 20 val steps

## Phase 1: Sequential LoRA TTT (COMPLETE)

Replaced per-sequence full-weight modification with per-sequence LoRA adapters on embedding-family params (embed, lm_head, ve0/1/2, bigram_embed).

### Results

| Config | val_loss | Δ vs no-TTT | val_time | step_avg | mem (MiB) |
|--------|----------|-------------|----------|----------|-----------|
| No TTT (pure eval) | 3.2826 | — | 17.5s | 0.9s | 34043 |
| Full-weight TTT (all Adam params) | 3.2774 | -0.0052 | 325s | 16.3s | 33933 |
| freeze_shared full-weight TTT | 3.2778 | -0.0048 | 302s | 15.1s | 34043 |
| Seq LoRA r=8, lr=1e-3, std=1/r, flat LR | 3.2818 | -0.0008 | 232s | 11.6s | 34043 |
| Seq LoRA r=128, lr=1e-3, std=1/r, flat LR | 3.2801 | -0.0025 | 249s | 12.4s | 34043 |
| Seq LoRA r=128, lr=1e-3, std=1, per-group LR | 3.2822 | -0.0004 | 250s | 12.5s | 34043 |
| **Seq LoRA r=128, lr=2.5e-4, std=1, per-group LR** | **3.2761** | **-0.0065** | **248s** | **12.4s** | **34043** |

### Key findings

1. **LoRA beats full-weight TTT**: best LoRA (3.2761) outperforms full-weight TTT (3.2774) by 0.0013, likely due to low-rank regularization.
2. **A init must be rank-independent** (`std=1`). Using `std=1/rank` kills learning at higher ranks.
3. **Per-parameter-group LRs are critical**: embed/lm_head use `base_lr` with `betas=(0.5, 0.95)`; ve/bigram use `base_lr * 75` with `betas=(0.75, 0.95)` — matching the training optimizer structure.
4. **LR must decrease with rank**: `lr=2.5e-4` for r=128 (effective output scales as `~lr * sqrt(rank)`).
5. **Sequential LoRA is 1.3× faster** than full-weight TTT (248s vs 325s) — no state_dict reload, fewer backward params.

### Architecture

- `EmbeddingLoRA`: `A = nn.Embedding(V, r)` (random init), `B = zeros(r, d)`. Output: `A(tokens) @ B`.
- `LinearLoRA`: `A = (r, d_in)` (kaiming init), `B = zeros(d_out, r)`. Output: `(x @ A.T) @ B.T`.
- `TTTLoRA` wraps embed_lora + 3 ve_loras + bigram_lora + lm_head_lora.
- Standalone `torch.optim.Adam` per-group, reset per sequence.
- LoRA attached to model as attribute → compiled into graph with `fullgraph=True`.

## Phase 2: Batched LoRA TTT (IN PROGRESS)

Goal: batch S sequences per forward pass to reduce kernel launch overhead and improve GPU utilization.

### Results so far

| Config | val_loss | Δ vs no-TTT | val_time | step_avg | steady-state | mem (MiB) |
|--------|----------|-------------|----------|----------|--------------|-----------|
| Batched r=8, S=8, lr=1e-3 (direct idx) | 3.2793 | -0.0033 | 237s | 11.9s | ~6s | 34043 |
| Batched r=128, S=32, lr=2.5e-4 (direct idx B gather) | 3.2762 | -0.0064 | 424s | 21.2s | ~30s | 41560 |
| Batched r=128, S=32, lr=2.5e-4 (loop inside compiled graph) | ~3.2770 | ~-0.006 | ~900s | ~45s | ~35s | 34043 |
| Batched r=128, S=32, lr=2.5e-4 (pre-computed deltas + seg_offsets) | 3.2761 | -0.0065 | 291s | 14.6s | ~8s | 34043 |
| Batched r=128, S=32, lr=2.5e-4 (pre-packed plans + pack_multiple=2048) | 3.2762 | -0.0064 | 263s | 13.2s | ~9.0s | 33933 |
| Batched r=128, S=32, lr=2.5e-4 (pre-packed + pack_multiple=2048 + no_grad eval deltas) | 3.2762 | -0.0064 | 258.6s | 12.9s | ~9.1s | 33933 |
| Batched r=128, S=32, lr=2.5e-4 (+ length-sorted sub-batches) | 3.2762 | -0.0064 | 176.5s | 8.8s | ~5.0s | 46805 |
| **Batched r=128, S=32, lr=2.5e-4 (+ merged eval+train fwd)** | **3.2767** | **-0.0059** | **199.8s** | **10.0s** | **~4.7s** | **46755** |

Steady-state = per-step time after compilation warmup (excluding recompile outlier steps).

No-TTT baseline re-checked on current code: `resume val_loss:3.2826 val_time:18001ms`, which is still about `0.90s/step` for 20 validation steps.

### What happened

**Quality matches sequential exactly** — batched produces identical val_loss (3.2761) at same hyperparameters.

**Iteration history:**
1. **Direct indexing** (`B[segment_ids]`): creates (T, r, d) intermediate per embedding LoRA call. At r=128, T=50K: ~10GB per call → memory thrashing (424s, +7.5GB peak memory).
2. **Per-segment loop inside compiled graph** with `@torch.compiler.disable`: graph breaks at every LoRA call (6× per forward), compiler can't optimize through them → very slow (~900s).
3. **Pre-computed deltas + seg_offsets**: embedding LoRA computed eagerly outside compiled graph with vectorized A lookup + contiguous B slice loop. Only lm_head LoRA causes one graph break → 291s total, **~8s/step steady-state** after warmup.
4. **Pre-packed chunk plans + bucketed packed lengths**: moved chunk planning and padding decisions out of the hot loop and reduced shape churn. This cut total time from 291s to 263s while preserving quality.
5. **No-grad eval LoRA deltas**: the eval-only embedding-family delta path was still building autograd graphs. Wrapping it in `torch.no_grad()` cut total time again from 263s to 258.6s.
6. **Length-sorted sub-batches**: sorting sequences by length before forming sub-batches groups similar-length sequences together, dramatically reducing padding waste and making packed lengths more uniform. This single-line change cut total time from 258.6s to **176.5s** and steady-state from ~9.1s to **~5.0s/step** — nearly 2× faster. Memory rose to 46805 MiB (from 33933) due to more uniform/larger packed tensors.
7. **Merged eval+train forward passes**: for non-final chunks, the eval and train process identical tokens with the same loss mask (since `train_start=0`). By extracting the eval loss from the training forward pass, we eliminate one full no_grad forward per non-final chunk. Steady-state improved from ~5.0s to **~4.7s/step** (~6%).

### Current speed picture

Best accurate sequential LoRA: `10.4s/step` steady-state.
Best accurate batched LoRA (merged eval+train): **`4.7s/step`** steady-state.
No-TTT pure eval: `0.9s/step`.

Current slowdown vs no-TTT: **5.2×**. Our first-principles estimate of the theoretical floor (unavoidable prefix re-evaluation + training passes) was **5-6×**. We are at the theoretical floor for exact LoRA TTT with chunk_size=512.

### Experiments that didn't help

- **S=64 sub-batch**: slower (6.4s/step) and 70 GiB memory. S=32 is optimal.
- **Truncated eval prefix (eval_ctx=1024)**: 4.6s/step but val_loss 3.2820 — lost 91% of TTT benefit. Layer 6 has full causal attention and needs the entire prefix.
- **Reduced training window (ttt_bs=512)**: 4.5s/step but val_loss 3.2793 — lost 48% of TTT benefit.
- **Reduced training window (ttt_bs=1024)**: 4.6s/step, val_loss 3.2771 — marginal savings, not worth quality trade-off.

### Speed optimization summary

| Stage | Steady-state | vs no-TTT | Speedup from original |
|-------|-------------|-----------|----------------------|
| Full-weight TTT (original) | 16.3s | 18.1× | 1.0× |
| Sequential LoRA TTT | 10.4s | 11.6× | 1.6× |
| Batched LoRA (pre-packed, no_grad deltas) | 9.1s | 10.1× | 1.8× |
| + Length-sorted sub-batches | 5.0s | 5.6× | 3.3× |
| + Merged eval+train forward | 4.7s | 5.2× | 3.5× |
| **Theoretical floor (exact TTT)** | **~4.5s** | **~5×** | **~3.6×** |

### Next steps

At 5.2× slowdown we are at the algorithmic floor for exact LoRA TTT. Remaining options:

1. **Approximate KV caching**: cache K,V tensors from previous forward passes; recompute only new chunk tokens through the transformer. Could break below 5× floor but requires significant model changes and may hurt accuracy.

2. **Phase 3 (TTT_LORA_PLAN.md)**: move on to integrating TTT into the training loop.
