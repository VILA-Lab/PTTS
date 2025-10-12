# P‑TTS: Prompting Test‑Time Scaling 🚀

**90 examples can beat 1K** — P‑TTS uses principled **instructional prompt augmentation** to turn **90 AIME seeds** into **900 high‑utility training examples**, delivering strong reasoning with far less data.

<p align="center">
  •
  📄 <a href="https://arxiv.org/abs/XXXX.XXXXX">Paper</a> •
  📊 <a href="https://huggingface.co/datasets/P-TTS/P_TTS-Full">Dataset</a>
</p>

<p align="center">
  <img src="image/Results.png" alt="P-TTS overview" width="80%">
</p>

## Table of Contents
- [What is P-TTS?](#what-is-p-tts)
- [Key Results](#key-results)
- [Training Data](#training-data)
- [Training](#training)
- [How It Works (pipeline)](#how-it-works-pipeline)
- [Reproduce](#reproduce)
- [Citation](#citation)

## What is P‑TTS?

P‑TTS expands a small, vetted seed set (90 AIME 2022–2024 problems) by **wrapping** each problem with *principled instructions* to elicit diverse reasoning traces from a teacher model (DeepSeek‑R1). We then fine‑tune Qwen2.5‑Instruct models on these augmented traces.

**Principles used (unchanged question text; wrappers are prefixed/suffixed):**

* **Reward** – e.g., "I'll tip \$200,000 for a better solution!"
* **Penalty** – "You will be penalized if the answer is wrong."
* **Correctness** – "You MUST provide the correct answer."
* **Step‑by‑Step** – "Think step by step."

> Data scales via augmentation multipliers m ∈ {1, 4, 5, 10}: **90 → 360 → 450 → 900**.

## Key Results

**Benchmarks:** AIME24, AIME25, MATH500, GPQA‑Diamond.
**Backbone:** Qwen2.5‑Instruct (7B/14B/32B).
**Metric:** accuracy (lm‑evaluation‑harness; greedy decoding).

| Model         | #Train ex. |     AIME24 |     AIME25 |    MATH500 |     GPQA‑D |       Avg. |
| ------------- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| **P‑TTS‑32B** |        900 | **73.33%** | **53.33%** | **94.20%** | **60.61%** | **70.35%** |
| **P‑TTS‑14B** |        900 |     53.33% |     26.67% |     90.40% |     51.01% |     55.35% |
| **P‑TTS‑7B**  |        900 |     43.33% |     26.67% |     84.20% |     41.92% |     49.03% |

## Training Data

The training dataset consists of 900 high-quality reasoning examples generated from 90 AIME seed problems. Each seed problem is augmented using principled instruction wrappers and processed through DeepSeek-R1 to create diverse reasoning traces.

**Dataset Composition:**
- **Source**: 90 AIME problems (2022-2024)
- **Augmentation**: 4 instruction wrapper types with reward variants
- **Final Size**: 900 training examples

### Data Tokenization

Before training, you need to tokenize your raw dataset. Use the provided tokenization script:

```bash
# Run the tokenization script
python tokenize_data.py
```


## Training

To run training, you can find our script at `train/sft.py` which you can invoke via one of the `train/sft*.sh` scripts, or you can launch via `train/launch.sh` if you are on a SLURM cluster (requires editing the file for your cluster setup).

### Configuration

**Hardware Requirements:**
- **For 7B models**: 4x A100 GPUs
- **For 32B models**: 6x B200 GPUs

**Quick Start:**
```bash
git clone https://github.com/simplescaling/s1.git
cd s1
pip3 install -r requirements.txt
# First tokenize your data
python tokenize_data.py
# Then run training
bash train/sft.sh
```
> Note: Training scripts are adapted from [simplescaling/s1](https://github.com/simplescaling/s1) (Apache-2.0).

### Training Data

The script expects your training data in CSV format. Update the `train_file_path` variable in `sft.sh`:
```bash
--train_file_path="xx_tokonized.csv"
```
---

## How It Works (pipeline)

```
90 AIME seeds → apply 4 instruction wrappers (+ reward variants) →
query teacher (DeepSeek‑R1) → collect reasoning traces → fine‑tune Qwen2.5‑Instruct
```

---

## Reproduce

```bash
# 1) Build wrapped prompts from seeds
python DataConstruction/build_prompt_variants.py \
  --input DataConstruction/seeds.csv \
  --out DataConstruction/variants.csv

# 2) Query teacher model to collect reasoning traces
python DataConstruction/deepseek_query.py \
  --input DataConstruction/variants.csv \
  --out DataConstruction/DS_responses.csv

# 3) Combine Full Data
python DataConstruction/combine_deepseek_data.py
```

---

## Citation
```
@article{bsharat2025ptts,
  title={Prompting Test-Time Scaling Is A Strong LLM Reasoning Data Augmentation -- 90 Samples Can Beat 1K in the Wild},
  author={Bsharat, Sondos Mahmoud and Shen, Zhiqiang},
  journal={arXiv preprint arXiv:2501.XXXXX},
  year={2025}
}
```
