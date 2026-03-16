# Batched Test-Time Training with Per-Sequence LoRA

## Problem

The current TTT implementation processes sequences **sequentially**: for each sequence in the validation batch, it resets the full model to `initial_model_state`, then iterates over chunks (eval chunk, train on chunk, advance). This is slow because:

1. **Full state_dict reload per sequence** (~300MB copy per sequence)
2. **Forward/backward through entire model** per chunk per sequence
3. **No cross-sequence parallelism** — GPU utilization is poor since each sequence is processed alone

Result: ~325s for 20 val steps on 2xH100. The speedrun submission was rejected for being too slow.

## Phase 1 Results: Sequential LoRA TTT (COMPLETE)

### Architecture
- `EmbeddingLoRA(vocab_size, dim, rank)`: `A` is `nn.Embedding(vocab_size, rank)`, `B` is `(rank, dim)` zeros. Output: `A(tokens) @ B`.
- `LinearLoRA(in_dim, out_dim, rank)`: `A` is `(rank, in_dim)`, `B` is `(out_dim, rank)` zeros. Output: `(x @ A.T) @ B.T`.
- `TTTLoRA` wraps: `embed_lora`, 3x `ve_loras`, `bigram_lora`, `lm_head_lora`.
- LoRA output added to base model output in `GPT.forward` (conditional on `getattr(self, 'ttt_lora', None)`).

### Key findings

**A initialization must be rank-independent (std=1)**. Using `std=1/rank` makes the effective gradient for B scale as `O(1/rank)`, causing higher ranks to learn slower — the opposite of intended behavior. With `std=1`, effective output LR scales as `~lr * sqrt(rank)`, so base_lr must decrease with rank.

**Per-parameter-group LRs matching the training optimizer are critical.** The training optimizer uses `lr_mul=75` for ve/bigram (effective LR=0.6) vs `lr_mul=1` for embed/lm_head (effective LR=0.008). Using a single flat LR for all LoRA params severely underperforms.

**Optimizer config**: `build_ttt_lora_optimizer()` creates Adam with:
- embed/lm_head groups: `lr=base_lr`, `betas=(0.5, 0.95)`
- ve/bigram groups: `lr=base_lr * 75`, `betas=(0.75, 0.95)`
- `eps=1e-10`

### Results

Checkpoint: `logs/ddf165fd-4ed9-43d8-aa3c-23332ae04564/state_pre_ttt.pt`

| Config | val_loss | Δ vs no-TTT | val_time | step_avg |
|--------|----------|-------------|----------|----------|
| No TTT (pure eval) | 3.2826 | — | 17.5s | 876ms |
| Full-weight TTT (baseline) | 3.2774 | -0.0052 | 325s | 16.3s |
| freeze_shared full-weight | 3.2778 | -0.0048 | ~285s | ~14.3s |
| LoRA r=8, lr=1e-3, std=1/r, flat LR | 3.2818 | -0.0008 | 232s | 11.6s |
| LoRA r=128, lr=1e-3, std=1/r, flat LR | 3.2801 | -0.0025 | 249s | 12.4s |
| LoRA r=128, lr=1e-3, std=1, per-group LR | 3.2822 | -0.0004 | 250s | 12.5s |
| **LoRA r=128, lr=2.5e-4, std=1, per-group LR** | **3.2761** | **-0.0065** | **248s** | **12.4s** |

**Best LoRA config beats full-weight TTT by 0.0013** (3.2761 vs 3.2774) while being ~1.3x faster (248s vs 325s). The quality gain likely comes from the LoRA's implicit low-rank regularization preventing overfitting during the few gradient steps per chunk.

### Phase 1 timing breakdown (per val_step avg)
- Sequential LoRA: **12.4s/step** (20 steps × 12.4s = 248s)
- Full-weight TTT: **16.3s/step** (20 steps × 16.3s = 325s)
- No-TTT eval: **0.9s/step** (20 steps × 0.9s = 17.5s)

The LoRA overhead vs no-TTT is ~11.5s/step. Most of this is the sequential per-sequence loop (forward+backward+optimizer per chunk). Batching should collapse this.

## Phase 2: Batched LoRA TTT

### Goal
Process S sequences simultaneously instead of sequentially. The base model forward is shared; only LoRA contributions differ per sequence.

### Current sequential flow (per val_step)
```
for each sequence (S ≈ 50 per GPU):
    reset LoRA + optimizer
    for each chunk (C ≈ 3-5 per sequence):
        eval: forward(prefix + chunk) with no_grad  → accumulate loss on chunk
        train: forward(context_window) → backward → optimizer.step on LoRA
```
Total forward passes: S × C × 2 ≈ 300-500, all sequential.

### Batched flow (target)
```
reset all S LoRAs + optimizers
for chunk_step in range(max_chunks):
    eval:  batched forward(all active sequences' prefixes+chunks) → accumulate losses
    train: batched forward(all active sequences' context windows) → backward → batched optimizer.step
    deactivate sequences that finished
```
Total forward passes: max_chunks × 2 ≈ 6-10, each processing all sequences in parallel.

### Key design decisions

**1. Batched LoRA storage**
Store all S sequences' LoRA params as stacked tensors:
- `A_embed`: `(S, vocab_size, r)` — per-sequence embedding A
- `B_embed`: `(S, r, dim)` — per-sequence projection B
- Same for ve0/1/2, bigram, lm_head

At r=128: per-sequence LoRA is ~65M params (~130MB bf16). For S=50: ~6.5GB. May need to reduce rank or batch fewer sequences.

**2. Batched LoRA forward**
The model uses packed varlen format (B=1, all tokens concatenated). We need a `segment_ids` tensor mapping each token to its sequence index.

For embedding LoRA:
```python
# tokens: (T,), segment_ids: (T,), A: (S, V, r), B: (S, r, d)
a = A[segment_ids, tokens]        # (T, r) — gather per-token
b = B[segment_ids]                # (T, r, d)
out = (a.unsqueeze(1) @ b).squeeze(1)  # (T, d)
```

For linear LoRA (lm_head):
```python
# x: (T, d), segment_ids: (T,), A: (S, r, d), B: (S, V, r)
a = A[segment_ids]                # (T, r, d)
b = B[segment_ids]                # (T, V, r)
out = ((x.unsqueeze(1) @ a.transpose(1,2)) @ b.transpose(1,2)).squeeze(1)  # (T, V)
```

**3. Gradient separation**
Since LoRA_i only affects tokens from sequence i (via segment_ids indexing), `loss = sum_i L_i` naturally gives independent gradients per sequence. No per-sample gradient tricks needed — standard backprop through the indexing ops.

**4. Batched optimizer**
Replace `torch.optim.Adam` with a manual batched Adam step on the stacked tensors. Each sequence has independent optimizer state (exp_avg, exp_avg_sq, step). All updated in parallel via element-wise ops on (S, ...) tensors.

**5. Chunking synchronization**
Different sequences have different lengths → different chunk counts. Strategy:
- Compute max_chunks across all sequences
- Maintain `active_mask: (S,)` — sequences that haven't finished
- For eval: only accumulate loss from active sequences' current chunks
- For train: mask out gradients from inactive sequences (or don't — they contribute 0 gradient since their tokens won't appear)

**6. Memory-efficient packing**
For each chunk step, pack all active sequences' tokens into a single flat tensor with `cu_seqlens` boundaries. This is what the model already expects — just with more "documents".

### Implementation steps

1. **Batched LoRA module**: `BatchedTTTLoRA(S, rank, model_dim, vocab_size, bigram_vocab_size)` storing stacked (S, ...) params
2. **Segment-aware forward**: modify `GPT.forward` to use segment_ids for batched LoRA lookup
3. **Batched optimizer**: manual Adam on stacked tensors
4. **`ttt_val_batch_batched()`**: new function implementing the batched chunking loop
5. **Wire up**: add env var flag to switch between sequential and batched

### Performance target
- No-TTT eval: 17.5s (20 val steps)
- Sequential LoRA: 248s
- **Batched LoRA target: 30-60s** (by collapsing S sequential passes into 1 batched pass)
- Bottleneck shifts to max_chunks × 2 forward passes, each ~2-3s (same as no-TTT but with LoRA overhead)

## Open Questions

1. **Memory budget at S=50, r=128**: ~6.5GB for LoRA params alone. May need r=32 or r=64 for batching. Need to validate quality at lower ranks with the tuned LR.
2. **torch.compile with segment_ids indexing**: the `A[segment_ids, tokens]` gather pattern needs to work with `fullgraph=True` and `dynamic=True`.
3. **Chunk alignment**: all sequences padded to same max_chunks — waste for short sequences. Could group by length.
4. **lm_head LoRA memory**: `B: (S, 50304, r)` at r=128, S=50 is 50×50304×128×2 = 645MB per lm_head alone. May need to fuse or use lower rank for lm_head.
