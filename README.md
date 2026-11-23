# Video Temporal Grounding: Video Swin Transformer vs Video ViT

A multimodal video temporal grounding implementation that localizes the time interval described by a natural-language query. The notebook implements and evaluates two alternative video encoders—**Video Swin Transformer** and **Video Vision Transformer (ViT)**—inside a shared language–video fusion and temporal span prediction pipeline on **Charades-STA**.

The repository artifact currently provided is a Jupyter notebook: `TEMPORAL_GROUNDING.ipynb`.

---

## 1. Problem Definition

Given an untrimmed video $V$ and a natural-language sentence $Q$, temporal grounding predicts the start and end timestamps of the video segment that semantically corresponds to the query:

$$(V,Q) \longrightarrow (\hat{t}_s,\hat{t}_e)$$

The predicted interval is evaluated against the annotated ground-truth interval using temporal Intersection over Union (IoU).

This is a **temporal localization / moment retrieval** task, not ordinary video classification: the model must identify *when* the described event occurs.

## 2. Experimental Objective

The central experiment compares two video representation strategies while keeping the downstream multimodal grounding pipeline substantially shared:

- **Video Swin Transformer:** hierarchical feature extraction with local shifted-window attention.
- **Video ViT:** 3D patch embedding followed by global self-attention.

Both backbones feed into the same text encoder, projection, cross-modal fusion, and learned-query span prediction design. This makes the experiment a backbone comparison within this implementation, although differences in parameter count, patch configuration, and compute mean it should not be interpreted as a perfectly capacity-matched architecture study.

## 3. Pipeline Overview

```text
Input video (16 RGB frames, 224 × 224)
                 |
        +--------+--------+
        |                 |
 Video Swin 3D          Video ViT
        |                 |
        +--------+--------+
                 |
          Video features
                 |
                 +-----------------------------+
                                               |
Natural-language query                         |
          |                                    |
   BERT-base-uncased                           |
          |                                    |
     Text features                             |
          |                                    |
          +----------+-------------------------+
                     |
       Residual projection / shared space
                     |
       Modality embeddings + positional encoding
                     |
       Bidirectional cross-modal fusion
                     |
       Learned-query temporal span head
                     |
       Candidate start/end spans + confidence
                     |
       Top-1 temporal localization / inference
```

### 3.1 Video encoders

| Component | Video Swin | Video ViT |
|---|---|---|
| Input | 16 frames, 224 × 224 | 16 frames, 224 × 224 |
| 3D patch size from active config | (2, 4, 4) | (2, 32, 32) |
| Embedding dimension | 96 at initial stage; hierarchical stages | 768 |
| Encoder depth | [2, 2, 6, 2] | 6 transformer blocks |
| Attention | Local shifted-window 3D attention | Global multi-head self-attention |
| Attention heads | [3, 6, 12, 24] | 12 |
| Drop path | 0.3 | 0.2 |
| Output feature dimension | 768 | 768 |

**Configuration note:** the notebook contains some older explanatory comments describing a ViT patch size of 16 and a 12-block ViT. The active `Config` and `build_video_vit(cfg)` settings use patch size `(2, 32, 32)` and depth `6`; this README documents the active configuration rather than those stale comments.

The Swin implementation includes a pretrained-weight loading utility with support for temporal inflation of 2D patch-embedding weights, checkpoint key-prefix handling, and graceful skipping of incompatible parameters. The active configuration sets `PRETRAINED_SWIN_PATH = None`, so a pretrained checkpoint is **not automatically loaded** unless the user supplies a valid path.

### 3.2 Text encoder

- Model: `bert-base-uncased`
- Maximum tokenized query length: 32
- Text feature dimension: 768
- The first 6 BERT layers are frozen by the active configuration.

### 3.3 Multimodal projection and fusion

The notebook implements:
- A three-stage residual projection into a shared feature space.
- Modality-type embeddings and sinusoidal positional encoding.
- Four bidirectional cross-modal fusion layers.
- Eight attention heads in the fusion module.
- Feed-forward dimension: 1024.
- Fusion dropout: 0.2.

Bidirectional fusion allows video and text tokens to exchange information without applying a causal attention mask.

### 3.4 Temporal span prediction

The prediction head uses five learned queries (DETR-style) to generate candidate temporal spans and confidence scores.

- Number of learned queries: 5
- MLP hidden dimensions: `[256, 128]`
- Confidence threshold used by inference: 0.5
- Temporal boundaries are represented in normalized video-time coordinates during training and converted to seconds during inference.

The training loss uses Hungarian matching to associate predicted spans with annotated ground-truth spans. The loss combines endpoint regression, temporal IoU, and confidence terms.

---

## 4. Dataset and Input Processing

The notebook uses the **Charades-STA** dataset with JSON annotation files and video files. The dataset loader expands each annotation into a video–sentence training example.

The saved notebook execution reports these loaded split sizes:

| Split | Video–query samples |
|---|---:|
| Train | 15,754 |
| Validation | 1,935 |
| Test | 1,961 |
| **Total** | **19,650** |

These are the counts reported by this notebook's local annotation files. They differ from commonly cited Charades-STA split statistics; verify the exact annotation files and split protocol before comparing these results with published benchmark numbers.

### Video preprocessing

1. Read the video using OpenCV.
2. Sample 16 frames uniformly across the video.
3. Resize frames to 224 × 224.
4. Convert BGR frames to RGB.
5. Apply clip-consistent training augmentation (random resized crop, horizontal flip, and color jitter).
6. Normalize using ImageNet channel statistics.

Validation and test frames are resized and normalized without the training augmentations.

### Text preprocessing

Queries are tokenized using the BERT tokenizer, padded or truncated to 32 tokens. The model receives `input_ids`, `attention_mask`, and `token_type_ids`.

### Expected dataset layout

The notebook sets the video directory to `./CHARADES_STA_DATASET/videos/videos/videos` and expects the annotation files below:

```text
project/
├── TEMPORAL_GROUNDING.ipynb
└── CHARADES_STA_DATASET/
    ├── Annotations/
    │   ├── train.json
    │   ├── val.json
    │   └── test.json
    └── videos/
        └── videos/
            └── videos/
                ├── <video_id>.mp4
                └── ...
```

The exact video filenames must match the `video_id` values in the JSON annotations.

---

## 5. Training Configuration

The active configuration and subsequent in-memory tuning patch define the following principal settings:

| Setting | Active value |
|---|---:|
| Frames per video | 16 |
| Frame resolution | 224 × 224 |
| Physical batch size | 8 |
| Gradient accumulation | 2 |
| Effective batch size | 16 |
| Maximum epochs in tuning patch | 26 |
| Swin learning rate | 2e-5 |
| BERT learning rate | 3e-5 |
| Head learning rate | 1e-5 |
| Weight decay | 0.05 |
| Warm-up epochs | 8 |
| Gradient clipping | 0.5 |
| Mixed precision | BF16 (CUDA) |
| Early-stopping patience | 10 |
| Early-stopping minimum delta | 0.1 |
| Evaluation IoU thresholds | 0.3, 0.5, 0.7 |

The tuning patch also changes the loss weights to:

$$\lambda_{L1}=8.0,\quad \lambda_{IoU}=3.0,\quad \lambda_{conf}=1.5$$

The training code includes checkpoint save/resume logic, optimizer and scheduler state handling, gradient accumulation, early stopping, and TensorBoard logging.

> **Reproducibility note:** the notebook has evolved through multiple configuration patches. Run the notebook from the beginning in order, and inspect the final active configuration before starting a new experiment. Existing checkpoint files can cause a run to resume rather than start from scratch.

---

## 6. Evaluation

The primary metric is **Recall@1 at temporal IoU threshold $m$**, written as $R@1, IoU=m$.

For predicted interval $[\hat{s},\hat{e}]$ and ground-truth interval $[s,e]$:

$$IoU = \frac{ \max(0,\min(\hat{e},e)-\max(\hat{s},s)) }{ \max(\hat{e},e)-\min(\hat{s},s) }$$

A query is counted as correct at threshold $m$ when the top-confidence predicted span has $IoU \ge m$.

The notebook evaluates:

- $R@1, IoU=0.3$
- $R@1, IoU=0.5$
- $R@1, IoU=0.7$

It also contains code for profiling parameter count, inference latency, GPU memory, epoch duration, and a ROC/AUC analysis. The ROC/AUC cell is present but has no saved completed output in the supplied notebook, so no ROC-AUC result is claimed here.

---

## 7. Saved Test Results

The notebook's saved test evaluation reports the following results:

| Test metric | Video Swin | Video ViT | Higher result |
|---|---:|---:|---|
| R@1, IoU = 0.3 | 56.25% | **57.93%** | ViT |
| R@1, IoU = 0.5 | 39.27% | **41.66%** | ViT |
| R@1, IoU = 0.7 | 19.79% | **22.13%** | ViT |

At the primary $IoU=0.5$ threshold, Video ViT is ahead by **2.39 percentage points**.

### Interpretation

- Video ViT achieves higher test recall at all three reported IoU thresholds.
- Video Swin has the higher best validation score in the saved comparison (37.83% versus 36.33%).
- The validation/test ranking difference is worth investigating; it may reflect generalization, checkpoint selection, or run-to-run variation.
- These values are the outputs saved in this notebook, not independently reproduced measurements.

---

## 8. Computational Profile

The notebook's saved profiling/dashboard output reports:

| Measurement | Video Swin | Video ViT | Lower / faster |
|---|---:|---:|---|
| Total parameters | 167.25 M | 186.95 M | Swin |
| Video encoder parameters | 27.85 M | 47.55 M | Swin |
| Single-sample inference latency | 24.30 ms | 19.53 ms | ViT |
| Derived inference throughput | 41.15 samples/s | 51.20 samples/s | ViT |
| Peak forward-pass VRAM | 1,751.0 MB | 1,703.4 MB | ViT |
| Mean epoch time | 12,597.9 s | 4,509.9 s | ViT |
| Mean epoch VRAM | 15,681.4 MB | 7,009.5 MB | ViT |
| Video-encoder GFLOPs | Not available | Not available | — |

The profiler did not produce valid GFLOPs values; the dashboard records zero because the profiling result was unavailable. Zero must **not** be interpreted as zero computational cost.

The latency-derived throughput is calculated as $1000 / latency_{ms}$. It is a profiler-derived estimate under the notebook's measurement setup, not a general deployment benchmark.

### Weighted comparison in the notebook

The notebook's own weighted scoring routine assigns:

- Video Swin: 3 points (18.8%)
- Video ViT: 13 points (81.2%)

Under that particular scoring scheme, **Video ViT is the overall winner**. This is a notebook-defined aggregate score, not a standard benchmark metric. Its result depends on the selected metrics and weights.

---

## 9. Installation and Execution

The notebook installs/imports its dependencies in its first cells. A typical environment for this implementation includes:

```bash
pip install torch torchvision transformers opencv-python numpy pandas matplotlib seaborn scipy scikit-learn tensorboard tqdm timm thop
```

Depending on the installed PyTorch build and hardware, install a CUDA-compatible PyTorch version separately by following the official PyTorch installation selector.

### Run

1. Place the Charades-STA videos and JSON annotations in the expected directory structure.
2. Open `TEMPORAL_GROUNDING.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab with suitable GPU access.
3. Verify the dataset paths and checkpoint paths in the configuration cell.
4. Execute the notebook cells in order.
5. Run the training/comparison cells to train or resume both backbones.
6. Run the test evaluation cell to produce test-set metrics.
7. Run the profiling and visualization cells to regenerate the comparison plots.

The notebook is configured for CUDA mixed precision with BF16. CPU execution may be substantially slower; the notebook's profiling and training paths should be checked if using a CPU-only environment.

### Checkpoints and logs

The configuration uses:

```text
checkpoints_v2/
logs_v2/
```

The experiment code stores separate checkpoint directories for the Swin and ViT runs. Checkpoints are used for resume and best-model evaluation.

---

## 10. Inference

The notebook includes a `predict(...)` function that accepts:

- A video file path.
- A natural-language query.
- A trained model.
- The BERT tokenizer.
- The active configuration and device.

It samples and normalizes video frames, tokenizes the query, runs the model, and returns candidate temporal spans in seconds.

Example query used in the notebook:

```python
DEMO_QUERY = (
    "The boxer in white trunks lands a series of punches on his opponent."
)
```

Set `DEMO_VIDEO` to a valid local video path before running the inference cell.

---

## 11. Visualizations and Analysis Included

The notebook includes code for:

- Swin and ViT model profiling.
- Training and validation metric comparison.
- Side-by-side test recall comparison.
- Sample prediction visualization with video frames and predicted temporal regions.
- Single-video natural-language query inference.
- Computational complexity visualization.
- ROC/AUC analysis at multiple IoU thresholds.

Some visualizations depend on executing the corresponding notebook cells and generating their runtime artifacts.

---

## 12. Limitations and Reproducibility Considerations

1. **Dataset split provenance:** the saved loader counts are 15,754 / 1,935 / 1,961. Confirm the annotation version and split protocol before comparing with external results.
2. **Backbone capacity is not matched:** the two encoders use different patch sizes and parameter counts. The comparison is useful for this pipeline but does not isolate attention type alone.
3. **Pretrained weights:** the active Swin pretrained path is `None`; do not describe the saved run as pretrained unless the actual checkpoint-loading log confirms it.
4. **Compute profiling:** GFLOPs were unavailable in the saved profiler output.
5. **ROC/AUC:** the ROC analysis code is included, but completed AUC values are not present in the saved output.
6. **Checkpoint state:** the notebook supports resuming from existing checkpoints. A clean rerun requires deliberate checkpoint management.
7. **Run variance:** report multiple seeds and confidence intervals before making strong claims about small performance differences.
8. **Benchmark comparability:** use the same dataset split, preprocessing, evaluation protocol, and checkpoint-selection policy when comparing against published methods.

---

## 13. Future Work

Potential extensions for a more rigorous study:

- Run at least three random seeds and report mean ± standard deviation.
- Match backbone parameter budgets or add controlled patch-size/depth ablations.
- Verify the official Charades-STA split and document the annotation source.
- Add a clean experiment configuration file and immutable run metadata.
- Report confidence intervals and per-query error analysis.
- Complete the ROC/AUC experiment and document the positive-label definition.
- Compare against established temporal grounding baselines under the same split and metric implementation.
- Evaluate robustness to longer videos, paraphrased queries, and temporally ambiguous actions.

---

## 14. Citation and Dataset Acknowledgement

This implementation uses the Charades-STA temporal sentence grounding dataset. Please cite the original Charades-STA dataset/method paper and follow the dataset's license and usage requirements when using or redistributing the data.

For background on the benchmark and task, see:

- [Charades-STA benchmark overview](https://www.sota2.com/research/dataset/charades-sta)
- [CVPR 2023 temporal sentence grounding paper discussing Charades-STA evaluation](https://openaccess.thecvf.com/content/CVPR2023/papers/Fang_You_Can_Ground_Earlier_Than_See_An_Effective_and_Efficient_CVPR_2023_paper.pdf)

---

## 15. Project Structure

```text
project/
├── TEMPORAL_GROUNDING.ipynb
├── CHARADES_STA_DATASET/
│   ├── Annotations/
│   │   ├── train.json
│   │   ├── val.json
│   │   └── test.json
│   └── videos/
│       └── videos/
│           └── videos/
├── checkpoints_v2/        # generated during training
└── logs_v2/               # generated during training
```

*The notebook is the authoritative implementation. This README documents its active configuration and saved outputs; it does not claim that every optional analysis cell has been executed successfully.*
