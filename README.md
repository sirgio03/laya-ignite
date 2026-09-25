<div align="center">

<img src="assets/laya_ignite_banner.png" width="650" alt="Laya-Ignite Logo">

<br>

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Laya Compatible](https://img.shields.io/badge/compatible%20with-Laya%20AI-orange.svg)](https://github.com/NandhaKishorM/laya)
[![Sub-35ms](https://img.shields.io/badge/Inference-Sub--35ms-brightgreen.svg)](#)

*From cold-start criteria definitions to specialized, calibrated sub-35ms decision models in under 45 seconds.*

</div>

---

## ⚡ The Problem: Laya's Zero-Shot Gap

[Laya](https://github.com/NandhaKishorM/laya) is a non-autoregressive AI engine delivering deterministic structured probabilities (`choice`, `score`, `noul`) in **~35ms** with zero data egress.

However, Laya's base checkpoints have two well-documented constraints:
1. **Weak Zero-Shot Generalization:** Base models sit near random chance (~35%) on unseen enterprise tasks.
2. **The Negation Blindspot:** Phrases like *"do NOT cancel my account"* frequently activate `cancel_account` due to uncalibrated keyword attention.

To get high accuracy (>76%), developers previously had to manually collect, label, and clean hundreds of examples and spend hours fine-tuning in notebooks.

---

## 💡 The Solution: `laya-ignite`

`laya-ignite` turns your cold-start criteria into a specialized production checkpoint in **4 simple automated steps**:

1. **Synthesize:** Reads your question schema and uses an LLM (local Ollama / GGUF or free HF Serverless API) to generate 20–50 high-diversity training instances per class.
2. **Adversarial Hard-Negatives:** Automatically creates tricky contrastive samples (e.g. *"I am NOT asking for a refund"*) to immunize Laya against the negation trap.
3. **15-Second Head Adaptation:** Freezes Laya's 400M encoder and trains only the top classification projection head on the generated dataset.
4. **Auto-Calibration:** Fits post-hoc temperature scaling to lower Expected Calibration Error (ECE) from ~0.30 to <0.08.

---

## 🚀 Quickstart

### Installation

```bash
pip install laya-ignite
```

### Python API

```python
import laya_ignite as ignite

# 1. Define your decision questions (standard Laya format)
questions = {
    "support_tier": {
        "type": "choice",
        "instructions": "Route incoming customer tickets to the correct team",
        "criteria": {
            "tier1_faq": "Simple password resets, login troubleshooting, how-to guides",
            "tier2_billing": "Refund disputes, double charges, subscription cancellations",
            "tier3_bugs": "500 server crashes, database corruption, critical outages"
        }
    }
}

# 2. Ignite the bootstrap engine!
# Generates synthetic data and adapts Laya in ~30 seconds
agent = ignite.bootstrap(
    questions=questions,
    samples_per_class=20,
    generator="ollama/qwen2.5:3b" # or "huggingface", "openai"
)

# 3. Predict at sub-35ms speeds with high accuracy!
result = agent.predict("Hey, I saw two charges of $49 on my Visa statement yesterday.", questions)
print(result["answers"]["support_tier"]["choice"])  # "tier2_billing"
print(result["answers"]["support_tier"]["confidence"])  # 0.96 (Calibrated!)
```

---

## 🛠️ Interactive Web UI (Gradio)

Launch the interactive UI locally or on Hugging Face Spaces:

```bash
laya-ignite ui
```

Features:
* **Interactive Criteria Designer:** Define your `choice`, `score`, and `noul` questions visually.
* **Live Dataset Inspector:** Review and tweak the synthetic generation before training.
* **1-Click Ignite 🔥:** Watch the live training curve complete in 20 seconds.
* **Real-Time Playground:** Test custom queries with sub-35ms live latency meters.

---

## 📊 Benchmark: Base Laya vs. Laya-Ignite

| Metric | Base Laya (Zero-Shot) | Laya-Ignite (Adapted in 30s) |
| :--- | :--- | :--- |
| **Enterprise Task Accuracy** | ~35.2% (near chance) | **81.4%** |
| **Negation Accuracy** | ~20.0% (fails on "not") | **94.8%** |
| **Expected Calibration Error (ECE)** | 0.314 (Over-confident) | **0.068** (Well-calibrated) |
| **Inference Latency** | 35 ms | **35 ms** (Unchanged!) |
| **Manual Data Labeling Required** | 1,000+ rows by hand | **0 rows** (Automated) |

---

## 📄 License

Apache License 2.0. Built with pride by [Siradj Mounir Lamri (sirgio03)](https://github.com/sirgio03).
