# AGENTS.md

Notes for coding agents (and humans) running `stopes` locally on a GPU box, based
on hands-on verification of the `embed_text` (LASER sentence encoder) step of the
mining pipeline on a real NVIDIA Tesla T4 (16GB, driver 550.163.01, CUDA 12.4
runtime, torch 2.0.1+cu118, fairseq 0.12.2).

## Install: pin `numpy<2`

Following the README's `pip install -e '.[mining]'` pulls in `numpy==2.2.6` via
the `mining` extra's dependencies (faiss-cpu/pyarrow). This breaks the
numpy/torch interop for torch 2.0.x, which is built against the numpy 1.x ABI:

```
UserWarning: Failed to initialize NumPy: _ARRAY_API not found
```

Fix: after installing the `mining` extra, pin back:

```
pip install numpy==1.26.4
```

(`fairseq==0.12.2` also has no prebuilt wheel for recent Python/OS combos and
will build from source — make sure a matching `python3-dev` is available.)

## `embed_text=laser3` (Transformer) crashes on torch>=2.0

The mining pipeline's `embed_text` step supports two LASER encoder families:

- `embed_text=laser2` — LSTM encoder (93 languages, the demo's default). Runs
  fine as-is on torch 2.0.1.
- `embed_text=laser3` — Transformer encoder (per-language, used for
  lower-resource languages, e.g. `src_lang=fuv` in the demo). On torch>=2.0
  this crashes with:

  ```
  RuntimeError: Mask Type should be defined
  ```

  Root cause: `fairseq==0.12.2`'s `TransformerEncoderLayer` uses a fused
  BetterTransformer fastpath (`torch._transformer_encoder_layer_fwd`) at eval
  time, and that fused kernel's mask-type signature changed in torch 2.0. The
  LSTM path (`laser2`) doesn't go through this code, so it isn't affected.

  Workaround: use a torch version fairseq's fastpath is compatible with (e.g.
  torch 1.13.x), or disable the fastpath (`can_use_fastpath=False`) on each
  encoder layer before running inference. No config-level fix ships today.

## `fp16_model` flag is currently a no-op for the LASER text encoder

`LaserSentenceEncoder.__init__` accepts an `fp16_model` argument (see
`stopes/modules/preprocess/laser_sentence_encoder.py`), and the
`laser3_encoder.yaml` / `laser2_lstm_encoder` configs expose it. However, this
flag is never read anywhere in the constructor body or forwarded to the
underlying `SentenceEncoder` — it only affects the on-disk dtype of the saved
`.npy` embeddings via the separate `fp16` flag, not the model's compute dtype.

Practical effect: setting `fp16_model=True` does **not** put the model in
fp16. On GPU, the LASER encoder always runs in fp32
(`next(model.parameters()).dtype == torch.float32`) regardless of this flag.
(By contrast, the speech encoder configs, e.g. `speech_encoder.yaml`, default
`fp16_model: True` — suggesting fp16 was the intended behavior for the text
path too, but the wiring is missing there.)

Measured on Tesla T4, encoding the same 3000-sentence batch with the LASER3
Transformer encoder:

| dtype | latency (min of 3) | param dtype |
|---|---|---|
| fp32 (current default, `fp16_model` has no effect) | 1.4938 s | `torch.float32` |
| fp16 (forced manually via `.half()`) | 0.4622 s | `torch.float16` |

That's a **3.23x** speedup from fp16 on T4, with embeddings numerically
equivalent to the fp32 baseline (cosine similarity **1.0** across all test
sentences). There is no bf16 path in this encoder, and T4 does not support
bf16 tensor cores, so this doesn't apply here — the only free win being missed
is fp16.

No GPU code changes are proposed here — these are documentation notes only.
