import argparse
import json
import os

import numpy as np
from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)


RESULTS_DIR = "results"


def load_ncbi_dataset():
    """Load the public NCBI Disease Corpus from Hugging Face."""
    try:
        return load_dataset("ncbi/ncbi_disease")
    except Exception:
        return load_dataset("ncbi_disease")


def tokenize_and_align_labels(examples, tokenizer, label_all_tokens=False):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)

            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics_builder(label_names):
    seqeval = evaluate.load("seqeval")

    def compute_metrics(predictions_and_labels):
        predictions, labels = predictions_and_labels
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_names[p] for p, l in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        true_labels = [
            [label_names[l] for p, l in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(
            predictions=true_predictions,
            references=true_labels,
        )

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    return compute_metrics


def build_training_args(args):
    """Construct TrainingArguments, handling the eval_strategy rename across
    transformers versions (evaluation_strategy was renamed to eval_strategy)."""
    common = dict(
        output_dir=os.path.join(RESULTS_DIR, "model"),
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir=os.path.join(RESULTS_DIR, "logs"),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
    )
    try:
        return TrainingArguments(eval_strategy="epoch", **common)
    except TypeError:
        # Older transformers versions use the previous parameter name.
        return TrainingArguments(evaluation_strategy="epoch", **common)


def save_results(trainer, tokenized_dataset, label_names, dataset, args, test_metrics):
    """Save human-readable results to results/ so they can be committed and viewed.

    Note: the model weights and logs under results/model and results/logs are
    intentionally git-ignored (they are large). The small files written here are
    what is meant to be committed to the repository.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Clean overall metrics (strip the "eval_" prefix the Trainer adds).
    overall = {
        k.replace("eval_", ""): float(v)
        for k, v in test_metrics.items()
        if isinstance(v, (int, float))
    }

    # Per-entity breakdown via a full seqeval report on the test set.
    seqeval = evaluate.load("seqeval")
    predictions, labels, _ = trainer.predict(tokenized_dataset["test"])
    preds = np.argmax(predictions, axis=2)
    true_preds = [
        [label_names[p] for p, l in zip(pr, la) if l != -100]
        for pr, la in zip(preds, labels)
    ]
    true_labels = [
        [label_names[l] for p, l in zip(pr, la) if l != -100]
        for pr, la in zip(preds, labels)
    ]
    full_report = seqeval.compute(predictions=true_preds, references=true_labels)
    full_report = {
        k: (v if not isinstance(v, dict) else {ik: float(iv) for ik, iv in v.items()})
        for k, v in full_report.items()
    }

    # 1) JSON with everything
    with open(os.path.join(RESULTS_DIR, "test_metrics.json"), "w") as f:
        json.dump(
            {"overall": overall, "per_entity": full_report},
            f,
            indent=2,
            default=float,
        )

    # 2) Simple CSV of the headline metrics
    with open(os.path.join(RESULTS_DIR, "test_metrics.csv"), "w") as f:
        f.write("metric,score\n")
        for key in ["precision", "recall", "f1", "accuracy"]:
            if key in overall:
                f.write(f"{key},{overall[key]:.4f}\n")

    # 3) Human-readable run summary
    with open(os.path.join(RESULTS_DIR, "run_summary.txt"), "w") as f:
        f.write("Disease NER - Run Summary\n")
        f.write("=========================\n\n")
        f.write(f"Model:        {args.model_name}\n")
        f.write(f"Epochs:       {args.epochs}\n")
        f.write(f"Batch size:   {args.batch_size}\n\n")
        f.write("Dataset splits (examples):\n")
        for split in dataset:
            f.write(f"  {split:12s}: {len(dataset[split])}\n")
        f.write(f"\nLabels: {label_names}\n\n")
        f.write("Test set metrics:\n")
        for key in ["precision", "recall", "f1", "accuracy"]:
            if key in overall:
                f.write(f"  {key:10s}: {overall[key]:.4f}\n")

    print(f"\nResults saved to: {os.path.abspath(RESULTS_DIR)}")
    print("  - test_metrics.json")
    print("  - test_metrics.csv")
    print("  - run_summary.txt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    dataset = load_ncbi_dataset()

    print("Dataset loaded:")
    print(dataset)

    sample = dataset["train"][0]

    if "tokens" not in sample:
        raise KeyError("Expected a 'tokens' column in the dataset.")

    if "ner_tags" not in sample and "tags" in sample:
        dataset = dataset.rename_column("tags", "ner_tags")

    label_feature = dataset["train"].features["ner_tags"].feature
    label_names = label_feature.names
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {label: i for i, label in enumerate(label_names)}

    print("\nLabels:")
    print(label_names)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    tokenized_dataset = dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer),
        batched=True,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = build_training_args(args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_builder(label_names),
    )

    print("\nTraining biomedical NER model...")
    trainer.train()

    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(tokenized_dataset["test"])
    print(test_metrics)

    save_results(trainer, tokenized_dataset, label_names, dataset, args, test_metrics)

    trainer.save_model(os.path.join(RESULTS_DIR, "model"))
    tokenizer.save_pretrained(os.path.join(RESULTS_DIR, "model"))


if __name__ == "__main__":
    main()
