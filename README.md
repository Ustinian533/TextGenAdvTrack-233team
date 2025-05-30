# TextGenAdvTrack-233team

This repository contains the codebase for **Team 233**'s solution in the *Text Generation Evasion and Detection Track* of the Spring 2025 UCAS AI Challenge.

We provide scripts for preprocessing, inference, and evaluation using a KNN-based retrieval framework. Our best-performing model weights are also available on Hugging Face.

## 🏆 Model Weights

The model checkpoints are publicly available at:  
👉 [https://huggingface.co/xin233/TextGenAdvTrack-233team](https://huggingface.co/xin233/TextGenAdvTrack-233team)

We recommend using `ucas_model2.pth` for optimal performance in evaluation.

---

## 📦 Data Preparation

All data files should be in `.jsonl` format. If your data is in `.csv` format, use the following command:

```bash
python script/read_csv.py
```

You can also convert the prediction output to `.csv` or `.xlsx` formats:

```bash
python script/jsonl2csv.py
python script/jsonl2xlsx.py
```

---

## 🧠 Embedding Database Generation

To build the few-shot database used in KNN-based retrieval:

```bash
python gen_emb.py \
    --path data/train_30shot.jsonl \
    --savedir savedir \
    --name database_name
```

* `--path`: Path to the few-shot `.jsonl` file.
* `--savedir`: Directory to save the database.
* `--name`: Name for the database file (final path: `savedir/database_name.pt`).

* The official database used in our submission is `data/train_30shot.jsonl`.
---

## 🚀 Inference

Run inference using the generated database:

```bash
python infer.py \
    --database_path savedir/database_name.pt \
    --test_dataset_path data/test.jsonl \
    --save_path results/predictions.jsonl \
    --maxK 5
```

* `--database_path`: Path to the `.pt` database file.
* `--test_dataset_path`: Path to the input test data in `.jsonl` format.
* `--save_path`: Where to save the predicted output.
* `--maxK`: The value of `K` for KNN retrieval.

## 🎯 Evasion Attacks

To apply text evasion attacks for adversarial evaluation, simply run the following three files in sequence.

```bash
python attackers/homoglyph.py
python attackers/upper_lower.py
python attackers/whitespace.py
```
