# Multicorn: Multimodal Enhancement of Single-Cell Hi-C Data

**Multicorn** (**Multi**modal Uni**corn**) is a multimodal blind super-resolution
framework that extends **ScUnicorn** by grounding single-cell Hi-C (scHi-C)
enhancement in the regulatory state of the genome. Where prior enhancers treat a
contact map as a self-contained image, Multicorn conditions reconstruction on
three orthogonal functional-genomics modalities, **ATAC-seq** (chromatin
accessibility), **ChIP-seq H3K27ac** (active enhancers), and **RNA-seq**
(transcriptional output), so that enhancement becomes a biology-guided inverse
problem rather than pixel-level restoration.

The enhanced contact maps are passed to **3DUnicorn** for 3D genome structure
reconstruction.

---

## Architecture

![Multicorn architecture](assets/multicorn_architecture.png)



## Folder Structure

```
Multicorn/
│
├── assets/
│   └── multicorn_architecture.svg     # Architecture diagram
│
├── configs/
│   └── default_config.yaml            # Model, objective, training, data settings
│
├── models/                            # Network components
│   ├── degradation_kernel.py          # Dynamic learned degradation operator T
│   ├── hic_encoder.py                 # 2D CNN encoder producing F_HiC
│   ├── modality_encoders.py           # ATAC / ChIP / RNA MLP encoders
│   ├── fusion.py                      # Multimodal fusion (z_multi, F_cond)
│   ├── restoration_loop.py            # Estimator + Restorer alternating loop
│   └── multicorn.py                   # Full Multicorn model
│
├── losses/                            # Biologically constrained objective
│   ├── biological_constraint.py       # B(H, O)
│   └── objective.py                   # J_Multi = data + β·P(H) + γ·B(H, O)
│
├── scripts/
│   ├── generate_multimodal_hr.py      # Enhance a dense map + omics → HR map
│   └── training/
│       ├── train.py                   # End-to-end training
│       └── infer.py                   # Tile inference + per-chromosome stitching
│
├── utils/
│   ├── preprocessing.py               # Hi-C normalization + omics bin alignment
│   ├── data_loader.py                 # Tile dataset with aligned omics tracks
│   ├── metrics.py                     # MSE, PSNR, SSIM, Pearson, Spearman
│   └── visualization.py               # Contact-map heatmaps and comparisons
│
├── data/
│   └── mouse_test_data/               # Pointer to shared ScUnicorn sample data
│
├── output/                            # Generated enhanced maps and figures
├── requirements.txt
└── README.md
```

---

## Installation

```bash
cd Multicorn
pip install -r requirements.txt
```

---

## Data

Multicorn reuses the matched mouse Islet-cell inputs shipped with `ScUnicorn`
(mouse chromosome 11 at 500 kb resolution). See
[`data/mouse_test_data/README.md`](data/mouse_test_data/README.md) for the file
list and GEO accessions (ATAC-seq `GSE160472`, H3K27ac ChIP-seq `GSE269897`,
RNA-seq `GSE287905`).

---

## Usage

### 1. Generate an enhanced HR contact map

From the `scripts/` directory:

```bash
cd scripts

python generate_multimodal_hr.py \
  --data_path ../../ScUnicorn/data/mouse_test_data/chr11_500kb.txt \
  --atac_data_path ../../ScUnicorn/data/mouse_test_data/GSE160472_ATAC_Seq.txt \
  --chip_data_path ../../ScUnicorn/data/mouse_test_data/GSE269897_CHIP_Seq.txt \
  --rna_data_path ../../ScUnicorn/data/mouse_test_data/GSE287905_RNA_Seq.txt \
  --output_image_path ../output/enhanced_chr11.png \
  --output_hic_path ../output/enhanced_chr11.txt \
  --model_path ../checkpoint/multicorn_model.pytorch
```

On Windows PowerShell, replace the trailing `\` line continuations with backticks
(`` ` ``). If `--model_path` is omitted or missing, the model runs with
initialized weights so the pipeline can be exercised without a checkpoint.

This writes an enhanced contact matrix (`output/enhanced_chr11.txt`) and a
heatmap (`output/enhanced_chr11.png`).

### 2. Train

```bash
cd scripts/training

python train.py \
  --train_data ../../data/train.npz \
  --atac_data ../../../ScUnicorn/data/mouse_test_data/GSE160472_ATAC_Seq.txt \
  --chip_data ../../../ScUnicorn/data/mouse_test_data/GSE269897_CHIP_Seq.txt \
  --rna_data ../../../ScUnicorn/data/mouse_test_data/GSE287905_RNA_Seq.txt \
  --epochs 50 --batch_size 64 --lr 0.0003 --gamma 0.1 --iterations 5
```

Omitting the three omics arguments trains the unimodal fallback (equivalent to
`γ = 0`).

### 3. Inference over a tile dataset

```bash
cd scripts/training

python infer.py \
  --input ../../data/test.npz \
  --checkpoint ../../checkpoint/multicorn_model.pytorch \
  --output ../../output/
```

### 4. 3D reconstruction

The enhanced matrix is consumed by **3DUnicorn**. Point its multimodal pipeline
at the Multicorn output:

```bash
cd ../3DUnicorn/src
python main_multimodal.py --parameters ../examples/parameters.txt
```

---

## Relationship to Unicorn

| Component | Role |
| --- | --- |
| **ScUnicorn** | Unimodal blind super-resolution of scHi-C contact maps |
| **Multicorn** | Multimodal extension that conditions enhancement on ATAC-seq, ChIP-seq H3K27ac, and RNA-seq |
| **3DUnicorn** | 3D genome structure reconstruction from enhanced maps |

The only architectural additions over the unimodal baseline are the three
modality encoders, the fusion injection, and the regulatory term `B(H, O)` in the
loss; the Deep Alternating Network backbone is shared.

---

## License

Released under the MIT License (see the repository root).
