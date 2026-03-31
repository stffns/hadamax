# hadamax

**Fast compressed approximate nearest-neighbor search.  Pure NumPy.  No heavy dependencies.**

`hadamax` implements the TurboQuant compression pipeline — randomized Hadamard transform followed by optimal Gaussian scalar quantization (Lloyd-Max) — as a self-contained Python library for embedding vector search.  It achieves **8–12× compression** with **>0.92 recall@10** against float32 brute-force, using only NumPy.

```
pip install hadamax
```

---

## Quick start

```python
import numpy as np
from hadamax import HadaMaxIndex

# Build index
idx = HadaMaxIndex(dim=384, bits=4)          # 4-bit, ~8x compression
idx.add_batch(ids=list(range(N)), vectors=embeddings)

# Query
results = idx.search(query_vector, k=10)     # [(id, score), ...]

# Persist
idx.save("my_index.hdmx")
idx2 = HadaMaxIndex.load("my_index.hdmx")   # atomic save, v1/v2 compatible
```

---

## Technical background

### The problem: embedding vectors are expensive

Modern embedding models produce float32 vectors of dimension $d \in \{384, 768, 1536\}$.  Storing $N$ vectors requires $4Nd$ bytes and brute-force search costs $O(Nd)$ per query.  For $N = 1\text{M}$, $d = 384$: **1.5 GB RAM**, and inner products dominate inference time.

Product Quantization (PQ) splits vectors into $M$ sub-vectors and quantizes each independently.  It is effective but requires training a $K$-means codebook, which is expensive and data-dependent.  Random Binary Quantization (RaBitQ, 1-bit) is fast but coarse.

**TurboQuant** (Zandieh et al., ICLR 2026, arXiv:2504.19874) achieves near-optimal distortion at $b$ bits per coordinate without training codebooks, by first rotating the space with a randomized Hadamard transform to make coordinates approximately Gaussian, then quantizing each coordinate independently with the optimal scalar quantizer for $\mathcal{N}(0,1)$.

---

### Algorithm

#### Step 1 — Normalize

Given a raw embedding $v \in \mathbb{R}^d$, compute the unit vector $\hat{v} = v / \|v\|$ and store $\|v\|$ separately (float32, 4 bytes).

#### Step 2 — Randomized Hadamard Transform (RHT)

Pad $\hat{v}$ to the next power of 2 ($d' = 2^{\lceil \log_2 d \rceil}$), then apply:

$$x = \frac{1}{\sqrt{d'}} \, H \, D \, \hat{v}$$

where $D = \text{diag}(\sigma_1, \ldots, \sigma_{d'})$ with $\sigma_i \stackrel{\text{i.i.d.}}{\sim} \pm 1$ (deterministic from a seed), and $H$ is the unnormalized Walsh-Hadamard matrix.

By the Johnson-Lindenstrauss lemma, each coordinate $x_i \approx \mathcal{N}(0, 1/d')$.  After rescaling $\tilde{x} = x \cdot \sqrt{d'}$, the coordinates are approximately $\mathcal{N}(0, 1)$ regardless of the original distribution of $v$.

**Complexity:** $O(d \log d)$ — no matrix multiplication.

#### Step 3 — Lloyd-Max scalar quantization

The optimal scalar quantizer for $\mathcal{N}(0,1)$ at $b$ bits partitions $\mathbb{R}$ into $2^b$ intervals and assigns each the conditional mean as reconstruction value.  These boundaries and centroids are precomputed and hardcoded in `hadamax._codebooks`:

| bits | levels | MSE distortion | bytes/coord |
|------|--------|----------------|-------------|
| 2    | 4      | 0.1175         | 0.25        |
| 3    | 8      | 0.0311         | 0.375       |
| 4    | 16     | 0.0077         | 0.5         |

The quantized vector is stored as a `uint8` index matrix, bit-packed to $b/8$ bytes per coordinate.

#### Step 4 — Approximate inner product

At search time, the query $q$ is rotated (not quantized) and the approximate cosine similarity is computed as:

$$\hat{\langle \hat{q}, \hat{v} \rangle} = \frac{1}{d'} \sum_{i=1}^{d'} c_{\tilde{q}_i} \cdot c_{\hat{v}_i}$$

where $c_j$ is the centroid for index $j$.  This is a single float16 matrix-vector product.

---

### TurboQuant_prod: unbiased estimator with QJL correction

The MSE quantizer introduces a systematic downward bias in the inner product estimate.  The `use_prod=True` mode corrects this using a **Quantized Johnson-Lindenstrauss (QJL)** residual:

1. Quantize at $(b-1)$ bits MSE, compute residual $r = \tilde{x} - \hat{x}_{\text{MSE}}$
2. Store $\text{sign}(S r)$ as a 1-bit vector (int8 in practice), where $S \in \mathbb{R}^{d' \times d'}$ is a fixed random Gaussian matrix
3. Store $\|r\| / \sqrt{d'}$ (one float32 per vector)

At query time, the correction term is:

$$\text{correction}_i = \frac{\sqrt{\pi/2}}{d'} \cdot \|r_i\| \cdot \bigl\langle S \hat{q}, \text{sign}(S r_i) \bigr\rangle$$

This follows from Lemma 4 of Zandieh et al. (2025): $\mathbb{E}[\text{sign}(S r) \mid r] = \sqrt{2/\pi} \cdot Sr / \|Sr\|$, which gives an unbiased estimate of $\langle r, \hat{q} \rangle$ up to a correction involving $\|r\|$.

**When to use `use_prod`:**
- When you need accurate inner product magnitudes (e.g., KV-cache compression, attention approximation)
- **Not** recommended for pure ranking/NNS — the added QJL variance degrades recall@k relative to MSE-only at equal total bits

---

### Compression ratios

For $N$ vectors of dimension $d = 384$:

| Backend       | Bytes/vector  | Ratio vs float32 |
|---------------|---------------|------------------|
| float32       | 1536          | 1.0×             |
| 4-bit hadamax | 192 + 4       | **7.9×**         |
| 3-bit hadamax | 144 + 4       | **10.4×**        |
| 2-bit hadamax | 96 + 4        | **15.4×**        |
| int8 (naïve)  | 384 + 4       | 3.9×             |

The 4-byte overhead is the per-vector norm.  The float16 search cache (centroid expansions) is allocated lazily and evicted on writes.

---

### Recall benchmarks

Measured on synthetic unit-sphere vectors ($d=384$, $N=10{,}000$, 100 queries).  **Baseline: exact cosine float32 brute-force.**

| bits | recall@1 | recall@10 | recall@50 |
|------|----------|-----------|-----------|
| 2    | 0.72     | 0.83      | 0.91      |
| 3    | 0.81     | 0.91      | 0.96      |
| 4    | 0.86     | 0.93      | 0.95      |

Recall improves with clustered (real-world) data.  On BGE-small-en embeddings from mixed document corpora, 4-bit achieves **recall@10 ≈ 0.95**.

> **Note on published results:** The TurboQuant paper (Zandieh et al., 2025) reports recall up to 0.99, measured against HNSW graph navigation (not brute-force float32), on GloVe $d=200$ data, using recall@1 with large $k_{\text{probe}}$.  These conditions differ from the above; both results are correct under their respective definitions.

---

### File format

Files use the `.hdmx` extension.

```
magic:   4 bytes  "HDMX"
version: 4 bytes  uint32 (1 or 2)
dim:     4 bytes  uint32  — original embedding dimension
bits:    4 bytes  uint32  — total bits (2, 3, or 4)
seed:    4 bytes  uint32  — rotation seed
n:       4 bytes  uint32  — number of vectors
flags:   4 bytes  uint32  — bit-0: use_prod  [v2 only]
─────────────────────────────────────────
packed_len: 4 bytes  uint32
indices:    packed_len bytes  — bit-packed uint8 MSE indices
norms:      n × 4 bytes  float32  — per-vector original norms
[prod only]
  qjl_signs:  n × pdim bytes  int8  — sign(S·r) per vector
  rnorms:      n × 4 bytes  float32  — ‖r‖/√d per vector
ids:    n × (2 + len) bytes  — uint16 length-prefixed UTF-8 strings
```

Saves are **atomic** on POSIX: writes to `.hdmx.tmp` then `os.replace()`.

---

## API reference

### `HadaMaxIndex(dim, bits=4, seed=0, use_prod=False)`

| Parameter  | Type  | Default | Description |
|------------|-------|---------|-------------|
| `dim`      | int   | —       | Embedding dimension |
| `bits`     | int   | 4       | Bits per coordinate (2, 3, or 4) |
| `seed`     | int   | 0       | Rotation seed — must match at build and query time |
| `use_prod` | bool  | False   | Enable QJL unbiased estimator (requires bits ≥ 3) |

### Methods

```python
idx.add(id, vector)                    # Add one vector
idx.add_batch(ids, vectors)            # Add N vectors (50x faster than loop)
idx.delete(id) -> bool                 # Remove by id, O(1) lookup
idx.search(query, k=10) -> list        # [(id, score), ...] descending
idx.save(path)                         # Atomic binary save
HadaMaxIndex.load(path)                # Load from .hdmx file
idx.stats() -> dict                    # Compression / memory info
len(idx)                               # Number of stored vectors
```

---

## Relation to TurboQuant / PolarQuant

`hadamax` implements the core compression pipeline from:

> Zandieh, A., Daliri, M., Hadian, A., & Mirrokni, V. (2025).
> **TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.**
> *ICLR 2026.*  arXiv:2504.19874.

The same algorithm was published concurrently under the name "PolarQuant" at AISTATS 2026.  Both names were already taken as PyPI packages; `hadamax` = **Hada**mard + Lloyd-**Max**, named after its two core operations.

The key contributions of `hadamax` over the reference paper code:
- **No scipy**: codebooks are hardcoded, eliminating the only heavy dependency
- **Batch WHT**: single $O(n \cdot d \log d)$ call for bulk inserts
- **Float16 cache**: centroid expansions stored in half precision, ~2× faster matmul on modern hardware
- **O(1) delete** via `_id_to_pos` dict + position compaction
- **Atomic persistence**: `.hdmx.tmp` → `os.replace()` pattern
- **Backward-compatible file format**: v1/v2 both loadable

---

## Installation

```bash
pip install hadamax
```

**Requirements:** Python ≥ 3.10, NumPy ≥ 1.24.  No other runtime dependencies.

For development:
```bash
git clone https://github.com/stffns/hadamax
cd hadamax
pip install -e ".[dev]"
pytest tests/ -v
```

---

## License

MIT © 2025 Jayson Steffens.  See [LICENSE](LICENSE).

The TurboQuant algorithm is described in arXiv:2504.19874 by Zandieh et al. (Google Research / ICLR 2026).  This package is an independent implementation.
