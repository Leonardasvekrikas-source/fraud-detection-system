# docs/ — source materials & write-ups

This folder holds the materials the implementation is grounded in.

## Place here
- `source-materials/` — the MSc thesis manuscript (`Leonardas_Vekrikas_DISVfm-24.pdf`) and the
  supporting method/results write-ups (baseline replication, feature-level fusion, XAI analysis).

## Citation note (important)
The source paper is:

> **Yousefimehr, B. & Ghatee, M. (2025).** *A distribution-preserving method for resampling
> combined with LightGBM-LSTM for sequence-wise fraud detection in credit card transactions.*
> *Expert Systems with Applications*, **262**, 125661. doi:10.1016/j.eswa.2024.125661

Some earlier draft write-ups mis-attributed this paper (to "Masalha, Baryannis & Vrana,
p. 125540"). That attribution is **incorrect** — use the citation above everywhere. Fix the
citation in any write-up before adding it to this folder.

## Result summaries (until Phase 2 moves them into tracked MLflow experiments)
- Replication on the European dataset — LightGBM F1 0.8503 / AUC 0.9851; LSTM F1 0.7723 / AUC 0.9525.
- Generalization on PaySim — LightGBM strongest (F1 0.8298 / AUC 0.9967).
- Fusion comparison — feature-level fusion did not beat decision-level (competitive AUC ≈ 0.97,
  but precision collapsed from meta-classifier miscalibration under class imbalance).
