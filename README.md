# Video Temporal Grounding: Video Swin Transformer vs Video ViT

A multimodal video temporal grounding implementation that localizes the time interval described by a natural-language query. The notebook implements and evaluates two alternative video encoders—**Video Swin Transformer** and **Video Vision Transformer (ViT)**—inside a shared language–video fusion and temporal span prediction pipeline on **Charades-STA**.

The repository artifact currently provided is a Jupyter notebook: `TEMPORAL_GROUNDING.ipynb`.

---

## 1. Problem Definition

Given an untrimmed video \(V\) and a natural-language sentence \(Q\), temporal grounding predicts the start and end timestamps of the video segment that semantically corresponds to the query:

\[
(V,Q) \longrightarrow (\hat{t}_s,\hat{t}_e)
\]

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
Natural-language query
          |
   BERT-base-uncased
          |
     Text features
          |
          +-------------------------+
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
