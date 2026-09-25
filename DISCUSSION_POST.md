# ⚡ Laya-Ignite: Bootstrapping System-1 Decision Intelligence in 30 Seconds

<div align="center">

<img src="https://raw.githubusercontent.com/sirgio03/laya-ignite/main/assets/laya_ignite_banner.png" width="750" alt="Laya-Ignite Banner">

<br>

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/sirgio03/laya-ignite/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Laya Compatible](https://img.shields.io/badge/compatible%20with-Laya%20AI-orange.svg)](https://github.com/NandhaKishorM/laya)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/sirgio03/laya-ignite/pulls)
[![Inference Sub-35ms](https://img.shields.io/badge/Inference-Sub--35ms-brightgreen.svg)](#)

*From cold-start criteria definitions to specialized, calibrated sub-35ms Laya decision heads in under 45 seconds.*

**[GitHub Repository: sirgio03/laya-ignite](https://github.com/sirgio03/laya-ignite)**

</div>

---

### 📖 Table of Contents
1. [The Story: Why I Built This](#1-the-story-why-i-built-this)
2. [The Core Philosophy: System-1 vs System-2 in AI](#2-the-core-philosophy-system-1-vs-system-2-in-ai)
3. [Tackling the Negation Blindspot (Issue #377)](#3-tackling-the-negation-blindspot-issue-377)
4. [Engineering & Architecture: How It Works in 30s](#4-engineering--architecture-how-it-works-in-30s)
5. [End-to-End Walkthrough (100% Local or Cloud)](#5-end-to-end-walkthrough-100-local-or-cloud)
6. [Community Hardware Benchmark Matrix (Call for Testers!)](#6-community-hardware-benchmark-matrix-call-for-testers)
7. [How to Contribute & Open PRs](#7-how-to-contribute--open-prs)
8. [Acknowledgments & Community](#8-acknowledgments--community)

---

### 1. The Story: Why I Built This

Over the last several weeks, I’ve had the privilege of contributing directly to Laya's core codebase — specifically building the high-throughput batch inference endpoint ([PR #387](https://github.com/NandhaKishorM/laya/pull/387)).

Working closely with Laya's architecture, I was blown away by its speed: delivering deterministic, non-autoregressive structured decisions (`choice`, `score`, `noul`) in **~35ms** with zero cloud egress is exactly what production systems need. 

However, when friends and I started trying to deploy Laya into real-world projects, we hit the **cold-start wall**:
* **The Annotation Burden:** To get good accuracy on custom business domains, you need hundreds of labeled examples. Most teams don't have labeled data ready.
* **The Keyword Attention Trap ([Issue #377](https://github.com/NandhaKishorM/laya/issues/377)):** Zero-shot models easily misinterpret negations (*"I am NOT asking for a refund"* gets mapped to *billing/refund* because of keyword activation).
* **The Calibration & Complexity Barrier ([Discussion #347](https://github.com/NandhaKishorM/laya/discussions/347)):** In Discussion #347, @Zei33 proved that when Laya is trained with an LLM teacher, it beats Jev on accuracy and runs at 30ms instead of 300ms — but concluded that 95% of people can't use it because manual training is too tedious and uncalibrated ($ECE \approx 0.25$, causing dangerous overconfidence).

**We built `laya-ignite` specifically to solve this:** It automates teacher bootstrapping in 30 seconds, injects contrastive hard-negatives for Issue #377, and fits post-hoc temperature scaling to bring $ECE < 0.08$ out of the box.

I am sharing this today because I believe in open-source collaboration, building in public, and learning from the community.

---

### 2. The Core Philosophy: System-1 vs System-2 in AI

In cognitive psychology (Daniel Kahneman's *Thinking, Fast and Slow*):
* **System-2:** Slow, deliberate, logical, compute-intensive (conscious reasoning).
* **System-1:** Instantaneous, instinctive, reflexive, automatic (<50ms).

In modern AI:
* Large Autoregressive LLMs (70B+, GPT-4, Claude) are **System-2**. They are great for writing essays or complex multi-step reasoning, but they make terrible reflex routers. Calling an 800ms API to decide *"is this ticket urgent?"* is architectural overkill.
* Laya is **System-1**: ultra-fast, deterministic, calibrated, edge-deployable.

#### The Laya-Ignite Thesis:
> **"Never run System-2 at runtime. Use System-2 once at build time to bootstrap your System-1 engine in 30 seconds."**

We use LLMs (either locally via [Ollama](https://ollama.ai) or via cloud APIs) **only to synthesize domain data and hard-negatives**. We then freeze Laya's 400M backbone, fit calibrated linear heads, and discard the heavy LLM entirely. Your production runtime remains pure sub-35ms System-1.

---

### 3. Tackling the Negation Blindspot (Issue #377)

Laya Issue [#377](https://github.com/NandhaKishorM/laya/issues/377) pointed out a critical failure mode: zero-shot classifiers frequently fail when input text contains semantic inversions.

#### The Concrete Failure Case:
```python
# Prompt: "I do NOT want to cancel my account, I just want to change my credit card."
# Base Laya Zero-Shot output:
{
    "action": "cancel_account",      # WRONG! Triggered by "cancel" + "account"
    "confidence": 0.89               # Over-confident!
}
```

#### How Laya-Ignite Fixes This:
During synthetic generation, Laya-Ignite activates an **Adversarial Contrastive Engine**. It automatically pairs positive criteria with synthetic contrastive hard-negatives:
1. **Direct negation:** *"I do not want X, please do Y."*
2. **Contrastive boundary:** *"While others might need X, I am strictly looking for Z."*
3. **Conditional negation:** *"Never trigger X unless Y happens."*

By training the linear heads against these contrastive boundaries, Laya learns true semantic polarity rather than keyword coincidence. In our tests ([`examples/02_anti_negation_demo.py`](https://github.com/sirgio03/laya-ignite/blob/main/examples/02_anti_negation_demo.py)), accuracy on negated edge cases improved from **~50% (random chance) to 100%** with zero latency penalty.

---

### 4. Engineering & Architecture: How It Works in 30s

```text
[ User Decision Schema: choice, score, noul ]
                      │
                      ▼
┌────────────────────────────────────────────────────────┐
│ 1. Multi-Provider Synthesizer Engine                   │
│    - 100% Local: Ollama (Qwen2.5, Llama3, Mistral)     │
│    - Cloud APIs: OpenAI, Groq, DeepSeek, OpenRouter, HF│
│    - Generates balanced domain samples                 │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│ 2. Adversarial Contrastive Engine (Issue #377 Fix)     │
│    - Injects hard-negatives & polarity flips           │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│ 3. Fast Frozen-Encoder Head Adaptation                 │
│    - 400M Laya encoder is 100% FROZEN (zero drift)     │
│    - Only trains lightweight linear projection heads   │
│    - Multi-task Cross-Entropy + MSE Loss (~20 seconds) │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│ 4. Post-Hoc Temperature Calibration                    │
│    - Optimizes temperature T: argmin NLL(z/T, y)       │
│    - Guarantees true probabilities: ECE < 0.08         │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
[ Production Artifact: Sub-35ms Calibrated Laya Model ]
```

#### Why only 20–30 seconds?
Because Laya's 400M encoder backbone is **completely frozen**, backpropagation only computes gradients through the top linear layers (<150k parameters). We extract encoder embeddings in batch, cache them, and optimize the heads with AdamW. It converges in a few dozen epochs on a laptop CPU!

---

### 5. End-to-End Walkthrough (100% Local or Cloud)

#### Installation
```bash
git clone https://github.com/sirgio03/laya-ignite.git
cd laya-ignite
pip install -e .
```

#### Python Example (Supporting all 3 Laya Primitives)

```python
from laya_ignite import Schema, choice, score, noul
from laya_ignite.synthesizer import OllamaProvider  # 100% offline & private
from laya_ignite.adapter import FastHeadAdapter

# 1. Define your multi-task decision schema
schema = Schema(
    department=choice("Route customer inquiries", {
        "billing": "Invoices, double charges, payment processing errors",
        "technical": "500 server crashes, database corruption, bugs",
        "sales": "Custom enterprise pricing, demo requests, seat upgrades"
    }),
    is_urgent=noul("Is there an ongoing catastrophic outage or revenue loss?"),
    sentiment=score("User frustration score", ["calm", "frustrated", "enraged"])
)

# 2. Synthesize domain training pairs (100% offline via Ollama)
# No API keys required, no data leaves your machine
synthesizer = OllamaProvider(model="qwen2.5:7b")
dataset = synthesizer.generate(schema, samples_per_class=25, adversarial_negations=True)

# 3. Fast head adaptation (15-30 seconds) + Temperature calibration
adapter = FastHeadAdapter(encoder_model="layaproject/laya-base")
calibrated_model = adapter.fit(dataset)

# 4. Save your production checkpoint
calibrated_model.save("./my-specialized-laya")

# 5. Run sub-35ms inference!
decision = calibrated_model.decide("Hey, our production database crashed and is throwing Error 500!")

print("Route:", decision.answers["department"].choice)          # "technical"
print("Confidence:", decision.answers["department"].confidence)  # 0.98 (Calibrated!)
print("Is Urgent:", decision.answers["is_urgent"].noul)          # 0.96
print("Frustration:", decision.answers["sentiment"].score)       # 1.85 ("frustrated")
```

---

### 6. Community Hardware Benchmark Matrix (Call for Testers!)

To make sure Laya-Ignite performs reliably everywhere, **my developer friends and I have started profiling it across our machines — but we want to build a real-world community benchmark matrix with YOU.**

We've written a dedicated, automated profiling script: [`benchmark_gpu.py`](https://github.com/sirgio03/laya-ignite/blob/main/benchmark_gpu.py).

#### How to run it:
```bash
python benchmark_gpu.py
```
*(The script measures device throughput, adaptation convergence time, single-query latency, batch latency, and post-hoc ECE.)*

#### The Community Matrix *(Active Testing — PRs & Replies Welcome!)*

| Device / Architecture | Compute Backend | Head Adaptation (50 samples/cls) | Inference Latency (Batch=1) | Batch=16 Latency | Calibrated ECE | Contributor / Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Intel Core i5-8250U** | PyTorch (CPU) | ~34.2s | 34.8 ms | 185.0 ms | 0.052 | ✅ Verified baseline (`@sirgio03`) |
| **NVIDIA GeForce RTX 4090** | CUDA | *~4.5s (target)* | *~6.2 ms (target)* | *~18.0 ms (target)* | *<0.05* | ⏳ **[PR / Data Welcome]** |
| **NVIDIA GeForce RTX 3080 / 4080** | CUDA | *~7.8s (target)* | *~9.5 ms (target)* | *~28.0 ms (target)* | *<0.05* | ⏳ **[PR / Data Welcome]** |
| **NVIDIA GeForce RTX 3060 / 4060** | CUDA | *~12.0s (target)* | *~14.0 ms (target)* | *~42.0 ms (target)* | *<0.06* | ⏳ **[Testing with friends]** |
| **Apple Silicon M3 / M4 Max** | MPS | *~8.5s (target)* | *~11.0 ms (target)* | *~32.0 ms (target)* | *<0.05* | ⏳ **[PR / Data Welcome]** |
| **Apple Silicon M1 / M2 (Base)** | MPS | *~18.0s (target)* | *~22.0 ms (target)* | *~68.0 ms (target)* | *<0.06* | ⏳ **[PR / Data Welcome]** |
| **NVIDIA A100 (80GB)** | CUDA / TensorRT | *~2.1s (target)* | *~3.8 ms (target)* | *~8.5 ms (target)* | *<0.04* | ⏳ **[PR / Data Welcome]** |

<details>
<summary>📋 <b>Click here for the template to submit your benchmark results!</b></summary>
<br>

Copy and paste this in a reply to this discussion, or open a PR updating the table in `README.md`:

```markdown
### Benchmark Submission
- **Device / GPU:** (e.g. RTX 4090 / M2 Pro / Ryzen 7)
- **OS:** (e.g. Ubuntu 22.04 / Windows 11 / macOS Sonoma)
- **PyTorch Device:** (cuda / mps / cpu)
- **Head Adaptation Time:** XX.X seconds
- **Batch=1 Latency:** XX.X ms
- **Batch=16 Latency:** XX.X ms
- **Calibrated ECE:** 0.0XX
- **GitHub Handle:** @yourusername
```
</details>

---

### 7. How to Contribute & Open PRs

I am actively working on this and want to learn alongside other contributors. Here are 4 specific areas where we’d love your contributions:

1. **Hardware Matrix Submissions:** Run `python benchmark_gpu.py` and open a PR adding your machine to the README table!
2. **Adversarial Linguistic Templates:** Expand `laya_ignite/synthesizer/` with trickier grammatical counter-examples (double negatives, sarcasm, slang, multilingual).
3. **New Synthesizer Backends:** Help add native integrations for local backends like `vLLM` and `llama.cpp`.
4. **Runtime Compilers:** Exporting trained Laya-Ignite heads to ONNX Runtime and TensorRT.

#### Quick Contribution Workflow:
```bash
# 1. Fork the repo and branch
git checkout -b feat/my-contribution

# 2. Run existing tests to ensure clean state
pytest tests/

# 3. Make your changes, commit, and push
git commit -m "feat: add adversarial template for conditional negations"
git push origin feat/my-contribution
```

---

### 8. Acknowledgments & Community

* A huge thank you to **[@NandhaKishorM](https://github.com/NandhaKishorM)** and the Laya contributors for designing such an elegant, ultra-fast System-1 architecture.
* Shoutout to my close developer friends who generously lent their hardware, caught edge cases, and helped run early benchmark passes!
* Built with pride by **[Siradj Mounir Lamri (@sirgio03)](https://github.com/sirgio03)**.

---

### 💬 Let's Discuss!
* What kinds of edge cases or classification tasks are you using Laya for?
* Would you find an ONNX / GGUF direct export valuable?
* If you run `benchmark_gpu.py`, please drop your numbers below so we can add you to the official matrix!

👉 **Repo Link:** **[https://github.com/sirgio03/laya-ignite](https://github.com/sirgio03/laya-ignite)**
