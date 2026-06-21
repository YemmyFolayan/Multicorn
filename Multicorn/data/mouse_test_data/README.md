# Mouse Test Data (mouse chromosome 11, 500 kb)

Multicorn reuses the same matched mouse Islet-cell inputs shipped with
`ScUnicorn`. To avoid duplicating large files, the sample data lives in
`../../../ScUnicorn/data/mouse_test_data/` and is referenced from there by the
scripts and the default config.

| Signal | File | GEO accession |
| --- | --- | --- |
| scHi-C contact map | `chr11_500kb.txt` | — |
| ATAC-seq (accessibility) | `GSE160472_ATAC_Seq.txt` | GSE160472 |
| H3K27ac ChIP-seq (active enhancers) | `GSE269897_CHIP_Seq.txt` | GSE269897 |
| RNA-seq (expression) | `GSE287905_RNA_Seq.txt` | GSE287905 |

To use a self-contained copy instead, place the four files in this directory and
update the paths in `configs/default_config.yaml`.
