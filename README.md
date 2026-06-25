# Multicorn: Multimodal Enhancement of Single-Cell Hi-C Data for Robust 3D Genome Structure Reconstruction

**Multicorn** (**Multi**modal Uni**corn**) is a multimodal blind super-resolution framework that grounds single-cell Hi-C (scHi-C) enhancement in the regulatory state of the genome. It is an improvement over **Unicorn**, integrating ATAC-seq, ChIP-seq H3K27ac, and RNA-seq as biological priors so that enhancement becomes a biology-guided inverse problem rather than pixel-level image restoration.

---

## Overview

The 3D organization of chromatin governs gene regulation, cellular identity, and disease, but reconstructing this architecture from scHi-C data is severely limited by extreme sparsity and stochastic noise. Existing enhancement methods, including recent blind super-resolution frameworks, operate on contact maps in isolation and therefore restore them as patterns of pixels, without regard for the underlying regulatory state of the genome. As a result, mathematically plausible contacts can still be biologically implausible.

Multicorn addresses this gap by integrating three orthogonal functional-genomics modalities:

- **ATAC-seq**: chromatin accessibility
- **ChIP-seq H3K27ac**: active enhancers
- **RNA-seq**: transcriptional output

A multimodal fusion layer encodes each 1D omics signal into a latent representation that conditions a deep alternating optimization loop, and a biologically constrained loss penalizes contacts that contradict the local regulatory landscape. Downstream 3D reconstructions generated from Multicorn-enhanced contact maps show improved agreement with orthogonal 3D-FISH measurements.

> Full implementation, runnable scripts, and detailed documentation live in [`Multicorn/`](Multicorn/).

---


## Architecture

![Multicorn architecture](Multicorn/assets/multicorn_architecture.png)

Multicorn extends the ScUnicorn blind super-resolution backbone with three additions over the unimodal baseline: independent encoders for the regulatory tracks, a fusion injection that conditions restoration on the regulatory context, and a biologically constrained objective. The Deep Alternating Network (DAN) backbone is shared.



## Installation

```bash
cd Multicorn
pip install -r requirements.txt
```

---

## Usage

### 1. Generate an enhanced HR contact map

```bash
cd Multicorn/scripts

python generate_multimodal_hr.py \
  --data_path ../../ScUnicorn/data/mouse_test_data/chr11_500kb.txt \
  --atac_data_path ../../ScUnicorn/data/mouse_test_data/GSE160472_ATAC_Seq.txt \
  --chip_data_path ../../ScUnicorn/data/mouse_test_data/GSE269897_CHIP_Seq.txt \
  --rna_data_path ../../ScUnicorn/data/mouse_test_data/GSE287905_RNA_Seq.txt \
  --output_image_path ../output/enhanced_chr11.png \
  --output_hic_path ../output/enhanced_chr11.txt \
  --model_path ../checkpoint/multicorn_model.pytorch
```

On Windows PowerShell, replace the trailing `\` line continuations with backticks (`` ` ``). If `--model_path` is omitted or missing, the model runs with initialized weights so the pipeline can be exercised without a checkpoint.

### 2. Train

```bash
cd Multicorn/scripts/training

python train.py \
  --train_data ../../data/train.npz \
  --atac_data ../../../ScUnicorn/data/mouse_test_data/GSE160472_ATAC_Seq.txt \
  --chip_data ../../../ScUnicorn/data/mouse_test_data/GSE269897_CHIP_Seq.txt \
  --rna_data ../../../ScUnicorn/data/mouse_test_data/GSE287905_RNA_Seq.txt \
  --epochs 50 --batch_size 64 --lr 0.0003 --gamma 0.1 --iterations 5
```

Omitting the three omics arguments trains the unimodal fallback (equivalent to `γ = 0`).

### 3. 3D reconstruction

The enhanced matrix is consumed by 3DUnicorn:

```bash
cd 3DUnicorn/src
python main_multimodal.py --parameters ../examples/parameters.txt
```

---

## Data

Multicorn uses matched mouse Islet-cell inputs on mouse chromosome 11 at 500 kb resolution. The omics tracks are obtained from NCBI GEO: ATAC-seq (`GSE160472`), H3K27ac ChIP-seq (`GSE269897`), and RNA-seq (`GSE287905`). See [`Multicorn/data/mouse_test_data/README.md`](Multicorn/data/mouse_test_data/README.md) for details.

---

## Repository Structure

```
Unicorn-Hi-C/
├── Multicorn/     # Multimodal (ATAC + ChIP + RNA) enhancement, primary framework
├── ScUnicorn/     # Unimodal blind super-resolution backbone Multicorn extends
└── 3DUnicorn/     # 3D genome structure reconstruction from enhanced maps
```

The end-to-end pipeline is: raw scHi-C `→` Multicorn enhancement `→` 3DUnicorn 3D reconstruction. `ScUnicorn` is the unimodal blind super-resolution backbone that Multicorn extends, and serves as the primary unimodal baseline.

---

## Documentation

See [`Multicorn/README.md`](Multicorn/README.md) for the full Multicorn documentation, including the architecture diagram, the biologically constrained objective, and per-component descriptions.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.
