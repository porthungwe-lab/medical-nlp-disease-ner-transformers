# Medical NLP with Transformers: Disease Named Entity Recognition

## Overview
This project demonstrates a biomedical NLP pipeline using Hugging Face Transformers for disease named entity recognition (NER).

The project uses the public **NCBI Disease Corpus**, an expert-annotated biomedical text dataset for disease mention recognition. The goal is to fine-tune a transformer-based token classification model to identify disease mentions in biomedical text.

This repository is designed as a compact research-portfolio project demonstrating:
- Biomedical NLP
- Transformer-based token classification
- Hugging Face datasets and models
- Evaluation for named entity recognition
- Relevance to clinical language models and medical AI research

## Clinical and Research Motivation
Clinical and biomedical documents often contain important disease-related information in unstructured text. Named entity recognition can support downstream applications such as:
- Clinical information extraction
- Patient record structuring
- Disease trajectory modelling
- Clinical decision support
- Semantic harmonization for medical AI systems

This project is not intended for clinical deployment. It is a small research prototype demonstrating how transformer models can be applied to biomedical text mining.

## Dataset
Dataset: **NCBI Disease Corpus**

The NCBI Disease Corpus is a public biomedical NLP dataset containing disease mentions annotated in biomedical literature.

Hugging Face dataset options:
- `ncbi/ncbi_disease`
- `ncbi_disease`

The training script first tries `ncbi/ncbi_disease` and falls back to `ncbi_disease` if needed.

Expected splits:
- train
- validation
- test

## Task
Token classification / Named Entity Recognition.

The model learns to classify each token as:
- outside a disease mention
- beginning of a disease mention
- inside a disease mention

The exact label names are loaded directly from the dataset.

## Model
Default model:

```text
distilbert-base-uncased
```

This is intentionally lightweight for quick experimentation. Future versions can replace it with biomedical models such as BioBERT, ClinicalBERT, PubMedBERT, or other domain-specific encoders.

## Repository Structure

```text
medical-nlp-disease-ner-transformers/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── train_ner.py
└── results/
```

## How to Run

```bash
pip install -r requirements.txt
python src/train_ner.py
```

Optional:

```bash
python src/train_ner.py --model_name distilbert-base-uncased --epochs 2
```

## Outputs
Running the script prints dataset information, the label mapping, and training/validation/test metrics. It also saves files to `results/`:
- `test_metrics.json` — overall and per-entity test metrics
- `test_metrics.csv` — headline precision / recall / F1 / accuracy
- `run_summary.txt` — model, epochs, dataset sizes, labels, and test scores
- `results/model/` — the fine-tuned model and tokenizer (git-ignored; too large to commit)

The model weights and training logs are intentionally **not** committed to the repository (see `.gitignore`). The small metrics files are what is meant to be committed and viewed here.

## Results
The model is trained by downloading the NCBI Disease Corpus and a transformer from Hugging Face, so results are produced when you run the script (not stored in the repo by default). After a run, fill in the table below from `results/test_metrics.csv`:
Corrected `distilbert-base-uncased` for 2 epochs on the NCBI Disease Corpus, evaluated on the held-out test set:

| Metric    | Score  |
|-----------|--------|
| Precision | 0.7825 |
| Recall    | 0.8469 |
| F1        | 0.8134 |
| Accuracy  | 0.9804 |

Recall (0.85) is higher than precision (0.78), meaning the model catches most disease mentions but over-predicts some, a reasonable trade-off for a lightweight, general-domain base model. Note that token-level accuracy (0.98) is inflated by the large number of non-entity ("O") tokens, which is why span-level F1 is the metric that actually reflects NER quality here. A biomedical encoder such as BioBERT or PubMedBERT would be the natural next step to push F1 higher.

*Full metrics, including the per-entity breakdown, are in `results/test_metrics.json`. Trained in ~90 seconds on a single Colab T4 GPU.*


## Relevance to Medical AI projects
This project is aligned with research areas such as:
- Clinical language models
- Biomedical NLP
- Medical information extraction
- Structured patient trajectory creation
- Explainable and interoperable medical AI
- Multimodal healthcare AI pipelines

It complements projects in PyTorch clinical risk prediction and patient survival modelling by demonstrating NLP skills relevant to clinical documentation.

## How I Built This
I built this as a compact, readable example of biomedical token classification rather than a maximal-performance system.

A few decisions to note:
- **Dataset.** I used the NCBI Disease Corpus because it is a public, expert-annotated benchmark for disease mention recognition, so the results are comparable to published work and nothing sensitive is involved.
- **Labels.** The label names (outside / beginning / inside a disease mention) are read directly from the dataset rather than hard-coded, so the script stays correct if the schema changes.
- **Tokenisation.** Transformer tokenisers split words into sub-word pieces, so I align the original word-level labels to the sub-word tokens and mask the extra pieces with `-100`, which tells the loss function to ignore them. This avoids counting a single word multiple times.
- **Model.** I defaulted to `distilbert-base-uncased` to keep training fast and accessible on modest hardware. It is swappable on the command line — biomedical encoders such as BioBERT, PubMedBERT, or ClinicalBERT would be the natural next step for higher accuracy.
- **Evaluation.** I used `seqeval`, the standard for NER, which scores at the level of whole entity spans rather than individual tokens — the metric that actually reflects how well disease mentions are extracted. I also save a per-entity breakdown, not just the overall score.
- **What gets committed.** Trained weights are large, so they are git-ignored; only the small, human-readable metrics files are committed. This keeps the repository lightweight while still showing the outcome.

## Limitations
- The dataset is based on biomedical literature, not hospital EHR notes.
- The default model is lightweight and not biomedical-domain-specific.
- Performance may improve with biomedical pretrained models and hyperparameter tuning.
- The project does not perform clinical validation.

## Future Work
- Fine-tune PubMedBERT or BioBERT
- Add entity-level error analysis
- Map extracted diseases to medical vocabularies
- Integrate extracted disease mentions into patient timeline modelling
- Add explainability methods for token-level predictions

## Author
Portia Hungwe  
M.Eng. Applied Artificial Intelligence  
BSc Statistics  
Healthcare and HealthTech AI Research Portfolio
