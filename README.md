<img width="223" height="96" alt="IEDD Logo" src="https://github.com/user-attachments/assets/056c9ac0-ce22-4549-9ec2-f5043c8d0ecc" />

# 🚗 IEDD: An Interaction-Enhanced Driving Dataset for Autonomous Driving

<img width="1107" height="353" alt="IEDD Overview" src="https://github.com/user-attachments/assets/72937afe-3362-4525-9570-a7b3caca8b74" />

This repository contains the official code for the paper **"IEDD: An Interaction-Enhanced Driving Dataset for Autonomous Driving"**.

<img width="1107" height="540" alt="IEDD Pipeline" src="https://github.com/user-attachments/assets/c92c68c4-7f9d-4311-a2ef-81abf7d2b5c1" />

---

## 📌 Contents

- [📕 Dataset Preparation](#-dataset-preparation)
- [🐍 Create Conda Environment (Python 310)](#-create-conda-environment-python-310)
- [📦 Install Dependencies](#-install-dependencies)
- [🎥 Generate Vision Clips + Action Semantics](#-generate-vision-clips--action-semantics-iedd-traj2visactpy)
- [🧩 Build ShareGPT QA Data (Q1–Q5)](#-build-sharegpt-qa-data-q1q5-iedd-2vqapy)
- [🔁 Add Counterfactual Q6 (IEDD-VQA_test)](#-add-counterfactual-q6-to-build-iedd-vqa_test-iedd-traj2q6py)
- [📊 Run Benchmark](#-run-benchmark-on-iedd-vqa_test-iedd-benchmarkpy)

---

## 📕 Dataset Preparation

### 1.1 Prepare trajdata cache (`root_dir/`)

For each supported raw dataset, follow the download instructions inside:

- `root_dir/<dataset_name>/readme.txt`

After downloading, preprocess the raw dataset into **trajdata cache** format and place the processed cache under `root_dir/`.

✅ Your final `root_dir/` should contain trajdata cache folders that `trajdata.UnifiedDataset(...)` can load.

---

### 1.2 Prepare IEDD / IEDD-VQA CSV annotations (`csv_dir/`)

Download the IEDD and IEDD-VQA CSV files from Zenodo:

- https://doi.org/10.5281/zenodo.18742437

Put the downloaded CSV files into `csv_dir/`.

> **Note**: The provided `csv_dir/` may already contain some IEDD CSV files, but you should **replace/complete** them with the full Zenodo release for full reproduction.

---

## 🐍 Create Conda Environment (Python 3.10)

We recommend **Python 3.10** for the IEDD environment.  
You may need to adjust the Python version depending on raw dataset preprocessing dependencies.

```bash
conda create -n IEDD python=3.10 -y
conda activate IEDD
```

---

## 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🎥 Generate Vision Clips + Action Semantics (`IEDD-traj2VisAct.py`)

This step consumes:
- trajdata cache in `root_dir/`
- CSV annotations in `csv_dir/`

and outputs:
- rendered **BEV videos** (vision part)
- extracted **action semantics** (`*.actions.json`)

### Example

```bash
python IEDD-traj2VisAct.py   --input-dir  csv_dir   --cache-root root_dir   --desired-data YOUR_DATASET_LABEL   --output-dir outputs/IEDD-VQA_vision   --timerange 10.0
```

**Important**
- `--desired-data` must match the dataset label used by trajdata in your cache.
- Action semantics JSON files are saved to:
  - `simulation_results/actions/*.actions.json` (by default)

---

## 🧩 Build ShareGPT QA Data (Q1–Q5) (`IEDD-2vqa.py`)

This step takes:
- the same `root_dir/` (to find `interaction.csv`)
- the `actions_dir` from Step 4
- (optional) the `videos_dir` from Step 4

and generates a **ShareGPT-format JSON** containing **Q1–Q5**.

### Example

```bash
python IEDD-2vqa.py   --root_dir    root_dir   --actions_dir simulation_results/actions   --videos_dir  outputs/IEDD-VQA_vision   --output_json outputs/IEDD-VQA_sharegpt_Q1Q5.json
```

---

## 🔁 Add Counterfactual Q6 to Build IEDD-VQA_test (`IEDD-traj2Q6.py`)

This step reads the Q1–Q5 JSON and appends **Q6 (counterfactual reasoning)** to build the **IEDD-VQA_test** JSON.

### Example

```bash
python IEDD-traj2Q6.py   --json-input  outputs/IEDD-VQA_sharegpt_Q1Q5.json   --json-output outputs/IEDD-VQA_test_Q1Q6.json   --input-dir   csv_dir   --cache-root  root_dir   --desired-data YOUR_DATASET_LABEL   --timerange   10.0
```

---

## 📊 Run Benchmark on IEDD-VQA_test (`IEDD-benchmark.py`)

Set your OpenRouter key (recommended via environment variable), then run evaluation on the generated test JSON.

```bash
export OPENROUTER_API_KEY="YOUR_KEY"
python IEDD-benchmark.py
```

Before running, update the following in `IEDD-benchmark.py`:
- `INPUT_JSON_PATH` → `outputs/IEDD-VQA_test_Q1Q6.json`
- `TARGET_MODELS` → the models you want to evaluate
- (optional) `RESULT_DIR`, `NUM_FRAMES`, etc.

---
