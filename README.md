# Bonsai vs Qwen3.5, on Edge

> **Note:** This benchmark is a quick experiment to compare these models on a Jetson Orin, not a thorough or rigorous evaluation. Take the results as rough directional signals, not definitive rankings.

> **Fairness note on Ternary-Bonsai speeds:** the `mlx-2bit` Ternary-Bonsai models are designed for Apple Silicon. To compare them head-to-head with the llama.cpp models on the same hardware, we ported them to run on Jetson CUDA (source build of MLX with sm_87 kernels). **The tok/s numbers reported here for Ternary-Bonsai are Jetson-MLX-CUDA numbers, not what you'd see on an M-series Mac or iPhone.** Per the model cards, the same weights run at ~30 tok/s on M4 Pro and ~100 tok/s on iPhone 17 Pro Max — several times faster than our Jetson port. Accuracy is intrinsic to the weights and is unaffected.

How good is the world's first 1-bit LLM, and can we stretch the same idea to a full family? We pit [Bonsai-8B](https://prismml.com/news/bonsai-8b) (1-bit, llama.cpp Q1_0) and the [Ternary-Bonsai collection](https://huggingface.co/collections/prism-ml/ternary-bonsai) (1.7B / 4B / 8B, 1.58-bit ternary weights served via MLX-CUDA) against six Qwen3.5 variants (0.8B–27B) on an NVIDIA Jetson Orin. All models answer the same 98 questions across 7 categories.

## About the Bonsai family

[Bonsai-8B](https://prismml.com/news/bonsai-8b) is the world's first commercially viable 1-bit LLM, developed by [PrismML](https://prismml.com/) — a startup that emerged from Caltech research with backing from Khosla Ventures, Cerberus, and Google. The entire network (embeddings, attention, MLP, LM head) is natively 1-bit, resulting in a 1.1 GiB model that is 14x smaller and 8x faster than a full-precision 8B model. It is released under the Apache 2.0 license.

The **Ternary-Bonsai** collection is a follow-up set of Apple-Silicon-first models packaged as `mlx-2bit` — their weights are ternary (values in {−1, 0, +1}, ~1.58 bits/weight) with 2.125 bits/weight effective storage in the MLX format. We run them on the Jetson's CUDA GPU via a source build of MLX with sm_87 kernels, served through a minimal OpenAI-compatible wrapper (`mlx_openai_server.py`).

The benchmark measures two things at once: how these aggressively-quantized Bonsai models fare against Qwen3.5's conventional Q4_K_M / Q4_K_S quantization, **and** how well a freshly-minted MLX CUDA backend performs on Jetson-class edge hardware.

## Models

| Model | Params | Quant | Runtime | Architecture | Weight Size |
|-------|-------:|-------|---------|--------------|------------:|
| **Qwen3.5-35B-A3B** | 35.5 B (3B active) | Q4_K_M | llama.cpp | MoE Hybrid SSM + Attention | 20.5 GiB |
| **Qwen3.5-27B** | 26.9 B | Q4_K_M | llama.cpp | Hybrid SSM + SWA + Full Attention | 15.6 GiB |
| **Qwen3.5-9B** | 8.95 B | Q4_K_M | llama.cpp | Hybrid SSM + Attention | 5.3 GiB |
| **Bonsai-8B** | 8.19 B | Q1_0 | llama.cpp | Dense Transformer (Qwen3-8B 1-bit) | 1.1 GiB |
| **Ternary-Bonsai-8B** | 8.19 B | mlx-2bit (ternary) | MLX-CUDA | Dense Transformer (Qwen3-8B 1.58-bit) | 2.1 GiB |
| **Qwen3.5-4B** | 4.21 B | Q4_K_M | llama.cpp | Hybrid Gated DeltaNet + Attention | 2.6 GiB |
| **Ternary-Bonsai-4B** | 4.02 B | mlx-2bit (ternary) | MLX-CUDA | Dense Transformer (Qwen3-4B 1.58-bit) | 1.1 GiB |
| **Qwen3.5-2B** | 1.89 B | Q4_K_M | llama.cpp | Hybrid Gated DeltaNet + Attention | 1.2 GiB |
| **Ternary-Bonsai-1.7B** | 1.72 B | mlx-2bit (ternary) | MLX-CUDA | Dense Transformer (Qwen3-1.7B 1.58-bit) | 462 MiB |
| **Qwen3.5-0.8B** | 0.82 B | Q4_K_S | llama.cpp | Hybrid Gated DeltaNet + Attention | 485 MiB |

The Qwen3.5 and original Bonsai-8B models are served via `llama-server` behind systemd units with flash attention enabled and thinking/reasoning disabled. The three Ternary-Bonsai models are served by a minimal MLX-CUDA OpenAI-compatible server (`mlx_openai_server.py`), also wired into systemd so the benchmark swaps them in and out the same way.

Per-model server configs:
[Qwen3.5-27B](qwen3.5-27b-server.md) | [Qwen3.5-9B](qwen3.5-9b-server.md) | [Qwen3.5-4B](qwen3.5-4b-server.md) | [Bonsai-8B](bonsai-8b-server.md)

## Benchmark Design

**98 questions** across **7 categories** and **3 difficulty levels** (easy / medium / hard):

| Category | Questions | Scoring |
|----------|:---------:|---------|
| General Knowledge | 14 | Exact match, keyword |
| Mathematics | 14 | Exact match |
| Coding | 14 | Execution-graded (Python test harnesses) |
| History | 14 | Exact match, keyword |
| Logical Reasoning | 14 | Exact match, constraint verifiers |
| Language Understanding | 14 | Exact match, keyword |
| Persian | 14 | Exact match, keyword |

Each question is run **3 times** per model. Scores report the mean across runs. Coding questions are graded by executing the model's output against a test suite (partial credit for passing some tests).

**Scripts:**
- `llm_benchmark.py` — runs the benchmark (manages systemd services, queries models, scores responses)
- `benchmark_eda.py` — generates analysis plots from the CSV results

## Results

**Dates:** 2026-04-01 (llama.cpp models) · 2026-04-17 (Ternary-Bonsai MLX-CUDA) | **Device:** Jetson Orin 30 GB

### Summary

![Summary Table](benchmark_plots/00_summary_table.png)

| Model | Accuracy | Gen tok/s | Prompt tok/s | Wall Time |
|-------|:--------:|:---------:|:------------:|:---------:|
| Qwen3.5-27B | **95.7%** | 9.5 | 107 | 444s |
| Qwen3.5-35B-A3B | 90.2% | 34.2 | 206 | 123s |
| Qwen3.5-9B | 90.2% | 27.0 | 320 | 167s |
| Qwen3.5-4B | 85.2% | 36.7 | 473 | 181s |
| Ternary-Bonsai-8B | 85.0% | 15.0 | 20 | 714s |
| Ternary-Bonsai-4B | 83.0% | 23.9 | 38 | 470s |
| Bonsai-8B | 78.9% | 46.5 | 554 | 117s |
| Qwen3.5-2B | 69.9% | 68.4 | 978 | 93s |
| Ternary-Bonsai-1.7B | 65.1% | 41.0 | 87 | 189s |
| Qwen3.5-0.8B | 53.4% | **100.9** | **1303** | **82s** |

### Overall Accuracy

![Overall Accuracy](benchmark_plots/01_overall_accuracy.png)

Qwen3.5-27B still leads at 95.7%. The 35B-A3B MoE ties the dense 9B at 90.2%. The new **Ternary-Bonsai-8B lands at 85.0%** — six points above the original Bonsai-8B (78.9%) and essentially level with the dense Qwen3.5-4B (85.2%) despite using half as much weight storage. **Ternary-Bonsai-4B (83.0%)** sits just under its Qwen counterpart with 40 % of the weight file. The smallest variant, **Ternary-Bonsai-1.7B (65.1%)**, slots between Qwen3.5-2B (69.9%) and Qwen3.5-0.8B (53.4%) — respectable given its 462 MiB footprint.

### Accuracy per GiB

![Accuracy per GiB](benchmark_plots/01b_accuracy_per_gib.png)

Weight-normalized, the small footprints dominate. Ternary-Bonsai-1.7B is the new leader at **1.44 accuracy/GiB** (0.65 / 0.451 GiB), edging out Qwen3.5-0.8B (1.13). Ternary-Bonsai-4B (0.79) beats the Qwen3.5-4B it targets. Bonsai-8B's Q1_0 (0.72) and Ternary-Bonsai-8B's mlx-2bit (0.40) both stay ahead of all 5 GiB+ models. The big takeaway: the Bonsai weight-compression story holds — even at 2.1 GiB, the 8B ternary model yields more accuracy per byte than Qwen3.5-9B or 27B.

### Accuracy by Category

![Category Accuracy](benchmark_plots/02_category_accuracy.png)

![Radar](benchmark_plots/04_radar_category.png)

**Strong across larger models:** General Knowledge, Coding, History, Language Understanding.

**Biggest differentiators:**
- **Math** — still the widest spread. Qwen3.5-27B hits 100%; the Ternary-Bonsai 4B and 8B both land at 78.6%, a clean +12 points over Bonsai-8B (66.7%). The 1.7B manages 64.3% — higher than Qwen3.5-2B's 31% — suggesting ternary quant preserves arithmetic better than the aggressive DeltaNet-hybrid compression at the 2B scale.
- **Logical Reasoning** — hardest category for every model. Ternary-Bonsai-8B reaches 71.4% (vs. Bonsai-8B's 55.9%), matching Qwen3.5-4B's 72.8%. The 1.7B stays at 39.2%, below the 0.8B→2B band.
- **Persian** — Qwen3.5-27B (91.7%) remains dominant. Ternary-Bonsai-8B (57.1%) edges Bonsai-8B (51.2%); multilingual is still the first casualty of extreme quantization at <9B.
- **Coding** — Bonsai-8B's 100% is now an outlier: Ternary-Bonsai-8B scores 92.9% and 4B scores 91.1%. The 1.7B drops to 56%. Code generation is robust down to ~4B but falls off fast below that.
- **General Knowledge** — all three Ternary models score **100%** on the 14-question GK set, matching the Qwen3.5-27B / 9B / 4B. Factual recall survives ternary quantization very well.

### Accuracy by Difficulty

![Difficulty Accuracy](benchmark_plots/03_difficulty_accuracy.png)

Top models handle easy questions (>90%). Hard questions expose the gap: Qwen3.5-27B stays above 90%, Ternary-Bonsai-8B and -4B hover near 75%, Bonsai-8B drops to ~73%, and the 0.8B / 1.7B fall to ~55%.

### Accuracy vs. Speed

![Accuracy vs Speed](benchmark_plots/06_accuracy_vs_speed.png)

The Ternary-Bonsai family shifts the Pareto frontier along the **accuracy axis**, not the throughput axis — on this Jetson with this MLX-CUDA build, they sit vertically under their Qwen counterparts rather than to the right of them. Ternary-Bonsai-8B matches Qwen3.5-4B in accuracy but generates at 15 tok/s vs. 36.7 tok/s; Ternary-Bonsai-4B trades roughly half the speed of Qwen3.5-4B for a similar score. This is an **implementation caveat, not a model property** — the same ternary weights on an M-series Mac run at 50–100+ tok/s.

### Speed Comparison

![Speed Comparison](benchmark_plots/05_speed_comparison.png)

The llama.cpp models still follow the memory-bandwidth rule: smaller footprint → faster gen. The MLX-CUDA Ternary-Bonsai models slot into the 15–41 tok/s band — decent for the Jetson's 205 GB/s memory, but far below what llama.cpp achieves for comparable weight sizes, and **dramatically** below what the same MLX models do on Apple Silicon. Prompt processing is the bigger gap: MLX on Jetson reports 20–87 prompt tok/s vs. llama.cpp's 107–1303, because our build's prefill kernels aren't taking advantage of batched attention. That's why Ternary-Bonsai-8B's full 98-question run takes 714 s vs. Bonsai-8B's 117 s — most of that is prompt prefill, not generation.

### Performance Details

![Wall Time](benchmark_plots/07_wall_time.png)

![Speed Distribution](benchmark_plots/09_speed_distribution.png)

![Speed by Difficulty](benchmark_plots/16_speed_by_difficulty.png)

### Scaling Analysis

![Accuracy vs Size](benchmark_plots/11_accuracy_vs_size.png)

![Efficiency](benchmark_plots/12_efficiency.png)

### Question-Level Analysis

![Question Heatmap](benchmark_plots/08_question_heatmap.png)

![Difficulty Category Heatmap](benchmark_plots/13_difficulty_category_heatmap.png)

![Model Agreement](benchmark_plots/14_model_agreement.png)

### Hardest Questions

![Hardest Questions](benchmark_plots/15_hardest_questions.png)

The hardest questions across all ten models remain the logic constraint puzzles (card ordering, clock angles, race ordering) and Persian language tasks. These require precise multi-step reasoning or strong multilingual knowledge — areas where smaller / more-quantized models struggle most. With 10 models now in the field, questions that the 27B aces but the 1.7B / 0.8B get wrong reveal the minimum model capacity required for each task.

### Verbosity

![Verbosity](benchmark_plots/10_verbosity.png)

## Key Takeaways

1. **Qwen3.5-27B is still the accuracy leader** at 95.7%. At 9.5 tok/s it's the slowest — best when correctness dominates latency.

2. **Ternary-Bonsai-8B is the standout new entry** — 85.0% accuracy closes most of the gap between Bonsai-8B (78.9%) and Qwen3.5-4B (85.2%). The biggest wins are math (+12) and logical reasoning (+16) vs. Bonsai-8B. Ternary weights clearly retain more reasoning capacity than Q1_0.

3. **Ternary-Bonsai-4B is the accuracy-per-byte winner above 1 GiB** — 83.0% from only 1.1 GiB, within 2 points of Qwen3.5-4B at 40% of the weight size.

4. **Ternary-Bonsai-1.7B is the new accuracy-per-byte champion overall** — 65.1% from 462 MiB. It beats Qwen3.5-0.8B by 12 points while being ~5% smaller on disk.

5. **MLX-CUDA on Jetson is much slower than llama.cpp** for the same-scale model. Prompt-processing throughput is the worst offender (20–87 vs. 107–1303 tok/s). The Ternary-Bonsai accuracy numbers are real; the throughput numbers reflect an early-stage MLX CUDA backend on sm_87, not the Bonsai family's inherent speed.

6. **Coding survives quantization down to ~4B** — Ternary-Bonsai-4B scores 91%, 8B scores 93%, Bonsai-8B scores 100%. Below that it collapses (1.7B=56%, 0.8B=44%).

7. **Persian / multilingual is still the first capability to degrade** — all sub-9B models are between 50% and 60% on Persian, regardless of quantization scheme.

8. **The original Bonsai-8B remains the latency champion** in its accuracy bracket. Via llama.cpp's Q1_0 it runs at 46.5 tok/s with 554 tok/s prompt — nothing else in the ≥78% accuracy group comes close on throughput on this hardware.

## Bonsai-8B vs Ternary-Bonsai-8B: Q1_0 llama.cpp vs mlx-2bit MLX-CUDA

Two ways to aggressively compress the same 8B Qwen3 architecture — which one works on the edge?

| | Bonsai-8B (Q1_0, llama.cpp) | Ternary-Bonsai-8B (mlx-2bit, MLX-CUDA) |
|---|:-:|:-:|
| Weight size | **1.1 GiB** (1-bit) | 2.1 GiB (~1.58-bit ternary, 2.13 bits/weight effective) |
| Overall accuracy | 78.9% | **85.0%** (+6.1 pts) |
| General Knowledge | 92.9% | 100% |
| Math | 66.7% | 78.6% |
| Coding | 100% | 92.9% |
| History | 96.4% | 98.2% |
| Logical Reasoning | 55.9% | 71.4% |
| Language Understanding | 89.3% | 96.4% |
| Persian | 51.2% | 57.1% |
| Gen tok/s (Jetson) | **46.5** | 15.0 |
| Prompt tok/s (Jetson) | **554** | 20 |
| Full-benchmark wall time | **117 s** | 714 s |

**Accuracy:** Ternary weights win almost everywhere. The big deltas are in reasoning-heavy categories — math (+12), logical reasoning (+16), language understanding (+7). The one category where Bonsai-8B's Q1_0 beats it is coding (100% → 92.9%), though that's a 1-question swing on a 14-question slice. Net: at a cost of 1 GiB more disk, Ternary-Bonsai-8B buys you roughly the accuracy of a dense Qwen3.5-4B.

**Throughput:** Bonsai-8B wins convincingly on this Jetson — 3× generation speed, 27× prompt speed, 6× faster end-to-end. But that's a runtime story (heavily optimized llama.cpp CUDA vs. a brand-new MLX CUDA backend on sm_87), not a model-architecture story. On Apple Silicon, the same `mlx-2bit` Ternary-Bonsai-8B is reported at ~30 tok/s on an M4 Pro; the Ternary-Bonsai-1.7B is listed at 103 tok/s on an iPhone 17 Pro Max. The ceiling on Jetson-MLX is an engineering problem, not a quantization one.

**Bottom line:** On today's Jetson, Bonsai-8B Q1_0 is the pragmatic choice if you need throughput and can live with ~79% accuracy. Ternary-Bonsai-8B is the pragmatic choice when accuracy matters more than tok/s, and becomes the dominant choice the moment the MLX-CUDA backend catches up to llama.cpp's kernel maturity. On Apple Silicon, Ternary-Bonsai is already the better option on all axes.

## Running

```bash
# Run the full benchmark (all 10 models)
uv run llm_benchmark.py

# Run specific models only
uv run llm_benchmark.py qwen3.5-2b qwen3.5-0.8b qwen3.5-35b-a3b
uv run llm_benchmark.py ternary-bonsai-1.7b ternary-bonsai-4b ternary-bonsai-8b

# Generate analysis plots
uv run benchmark_eda.py
```

Requires passwordless sudo for `systemctl start/stop llama-server-*` (see `/etc/sudoers.d/llama-benchmark`). The existing sudoers rule matches `llama-server-*`, which covers both the llama.cpp Qwen/Bonsai units and the new MLX `llama-server-ternary-bonsai-{1.7b,4b,8b}.service` units.

### MLX-CUDA setup on Jetson (for Ternary-Bonsai)

The three `mlx-2bit` Ternary-Bonsai models are served by `~/ai/test-mlx-on-cuda/mlx_openai_server.py`. The runtime needs a source-built MLX with sm_87 kernels, a newer cuDNN than the system one, and an env flag to disable cuDNN SDPA (which has no execution plan on sm_87). Once the `.venv` is set up, the systemd units pick it all up automatically. The key knobs:

- `MLX_CUDA_ARCHITECTURES=87-real` at build time
- `LD_LIBRARY_PATH=.../nvidia/cudnn/lib` (pip wheel, cuDNN 9.21+)
- `MLX_CUDA_USE_CUDNN_SDPA=0` (force non-cuDNN SDPA)

## Hardware

- **Device:** NVIDIA Jetson Orin
- **Memory:** 30,696 MiB unified (shared CPU/GPU)
- **CPU:** 12 threads (ARM Cortex-A78AE)
- **GPU:** Ampere (compute capability 8.7)
- **Memory Bandwidth:** ~205 GB/s
- **CUDA:** 12.6 · **Driver:** 540.4.0 · **cuDNN:** 9.3 (system) / 9.21 (pip wheel, used by MLX)
- **MLX:** 0.31.1 built from source with `MLX_BUILD_CUDA=ON MLX_CUDA_ARCHITECTURES=87-real`

## Author

Arman Jafarnezhad w/ Claude Opus 4.6 Max Effort · Ternary-Bonsai MLX-CUDA run: Claude Opus 4.7 (1M)

## Citation

If you use this benchmark or build on its results, please cite it:

```bibtex
@software{jafarnezhad_bonsai_vs_qwen_2026,
  author  = {Jafarnezhad, Arman},
  title   = {Bonsai vs Qwen3.5 on Edge: Benchmarking Aggressively Quantized LLMs on NVIDIA Jetson Orin},
  year    = {2026},
  url     = {https://github.com/ArmanJR/PrismML-Bonsai-vs-Qwen3.5-Benchmark},
  version = {2026.04.17}
}
```
