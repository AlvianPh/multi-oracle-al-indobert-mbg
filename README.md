# Adaptive Multi-Oracle Active Learning with IndoBERT for Indonesian Sentiment Classification

Code and data for the paper:

> Minarwati and A. P. Hardiadi, "Adaptive Multi-Oracle Active Learning using IndoBERT Representations for Efficient Indonesian Sentiment Classification," ICERA 2026 5th International Conference on Electronics Representation and Algorithm (IEEE).
> DOI: [10.1109/ICERA72709.2026.11666714](https://doi.org/10.1109/ICERA72709.2026.11666714)

Department of Informatics, STMIK El Rahma Yogyakarta.

This work extends our preliminary single-oracle study (SVM + Margin Sampling, TF-IDF; macro-F1 0.5389), published in *FAHMA* (Sinta 4): [10.61805/fahma.v24i2.203](https://doi.org/10.61805/fahma.v24i2.203).

## Overview

An Active Learning framework for classifying sentiment (Negative / Neutral / Positive) of YouTube comments about Indonesia's Free Nutritious Meal (MBG) program, designed to reduce annotation cost. Three label sources with different reliability are combined:

| Oracle | Label source | Sample weight |
|---|---|---|
| A | Human annotation (highest reliability) | 1.0 |
| B | Pseudo-labels from a frozen IndoBERT model (confidence ≥ 0.70) | 0.3 |
| C | IndoBERT fine-tuned iteratively during the AL process | 0.7 |

Samples are routed to an oracle by prediction entropy (high uncertainty → A, medium → C, low → B), so scarce human labels are spent only on the most ambiguous comments.

- **Features:** L2-normalised 768-d `[CLS]` embeddings from `Aardiiiiy/indobertweet-base-Indonesian-sentiment-analysis`
- **Classifier:** `LinearSVC` (balanced class weights) with cross-validated probability calibration
- **Query strategies:** Entropy, Least Confidence, Margin Sampling, K-Center Greedy, and Random Sampling as the passive baseline
- **Protocol:** seed set of 100 labels, 50 queries per iteration, budget of 1,500; three random seeds (42, 123, 456); evaluation on an independent test set of 700 human-annotated comments

## Key results

Macro-F1 on the 700-comment test set (mean ± std over 3 seeds, best strategy per configuration):

| Configuration | Best strategy | Macro-F1 |
|---|---|---|
| Multi-Oracle, weighted | Random | **0.6277 ± 0.0289** |
| Multi-Oracle, unweighted | Random | 0.6226 ± 0.0038 |
| Oracle B only | Margin Sampling | 0.5902 ± 0.0124 |
| Oracle A only | Margin Sampling | 0.5073 ± 0.0603 |

- **Label efficiency:** with Oracle B, Entropy and Margin Sampling reached 90% of the passive baseline's performance using only **149 labels (90.1% fewer than the 1,499 budget)**; in the multi-oracle setting Margin Sampling needed 199 labels (86.7%).
- **Oracle usage:** Oracle B ≈ 58–64%, Oracle C ≈ 33%, Oracle A ≈ 8% — human labels are reserved for the most uncertain samples.
- In the multi-oracle setting, uncertainty-based strategies did not consistently beat Random Sampling, suggesting that oracle diversity contributes more than the choice of query strategy.

## Data (`data/`)

| File | Rows | Description |
|---|---|---|
| `sample_999_labeled.csv` | 999 | Human-annotated comments. The notebook splits it (stratified, seed 42) into a 700-comment test set and a 299-comment pool used for Oracle A. Delimiter `;`, encoding latin-1. |
| `labeled_indobert_mbg.csv` | 7,967 | Unlabeled-pool comments with IndoBERT pseudo-labels and confidence (`bert_conf`). Delimiter `,`, UTF-8. |

Columns: `video_id`, `video_label`, `text` (raw comment), `votes`, `text_clean`, `text_processed`, `sentiment`, `label_enc` (0 = Negative, 1 = Neutral, 2 = Positive), plus `text_len` / `word_count` (human file) or `bert_conf` (pool file).

The comments are public YouTube comments about the MBG program, shared for research reproducibility. Usernames were removed and `@mentions` inside comment text were replaced with `@user`.

## Running the notebook

`AL_Multi_Oracle_MBG_IndoBERT.ipynb` runs top to bottom. Each experiment (Oracle B, Oracle A, Multi-Oracle weighted / unweighted) lives in its own cell and appends to `ALL_RESULTS`, so partial re-runs are possible.

- **Google Colab (recommended):** a GPU is needed for the multi-oracle runs (≈ 80 min each on an A100). Open the notebook with the repo's `data/` folder available, or place the two CSVs in `MyDrive/MBG_AL` (the notebook detects it).
- **Local:** `pip install -r requirements.txt`, then `jupyter notebook`. Outputs are written to `outputs/`.

## Reproducibility notes

- The saved notebook outputs come from the original Colab run (A100).
- The macro-F1 values, label-efficiency figures and the best-model classification report in the notebook match the paper. Accuracy / AULC and the oracle-usage percentages of the multi-oracle configurations may differ slightly from the printed tables.
- Fine-tuning on GPU is not bit-wise deterministic, so re-running can produce small differences even with fixed seeds.

## Citation

If you use this code or data, please cite the paper above.
